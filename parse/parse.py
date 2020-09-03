import os
import sys
import json
import time
import signal
import threading
import traceback
import queue as Queue
import multiprocessing
from multiprocessing.queues import JoinableQueue
from absl import app
from absl import flags
from future.builtins import range

import numpy as np
from scipy import sparse

from google.protobuf.json_format import MessageToJson

from pysc2 import run_configs
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb

from extract_global import GlobalParser
from extract_spatial import SpatialParser
from extract_actions import extract_actions

from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(name='library_path', default='../replay_library',
                    help='Directory replay list')
flags.DEFINE_string(name='output_path', default='../parsed_replays',
                    help='Path for saving results')
flags.DEFINE_string(name='player_race', default='Protoss',
                    help='Player race')
flags.DEFINE_string(name='enemy_race', default='Terran',
                    help='Enemy race')
flags.DEFINE_string(name='map', default = None,
                    help='Specific maps to parse, separated by ;')

flags.DEFINE_integer(name='n_instance', default=1,
                     help='# of processes to run')
flags.DEFINE_integer(name='batch_size', default=10,
                     help='# of replays to process in one iter')

flags.DEFINE_integer(name='step_size', default=72,
                     help='# of frames to step')

flags.DEFINE_integer(name='width', default=24,
                     help='World width of rendered area in screen')
flags.DEFINE_integer(name='map_size', default=64,
                     help='Spatial observation size in pixels')

FLAGS(sys.argv)
size = point.Point(FLAGS.map_size, FLAGS.map_size)
interface = sc_pb.InterfaceOptions(raw=True, score=True,
                feature_layer=sc_pb.SpatialCameraSetup(width=FLAGS.width, allow_cheating_layers=True),)
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)


class ReplayProcessor(multiprocessing.Process):
    """A Process that pulls replays and processes them."""
    def __init__(self, run_config, replay_queue):
        super(ReplayProcessor, self).__init__()
        self.run_config = run_config
        self.replay_queue = replay_queue

    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Kill thread upon termination signal
        while True:
            with self.run_config.start() as controller:
                for _ in range(FLAGS.batch_size):
                    try:
                        replay = self.replay_queue.get()
                    except Queue.Empty:
                        return
                    try:
                        if not os.path.isfile(replay['replay_path']): # Unable to find replay
                            print('Unable to locate', replay['replay_path'])
                            return

                        replay_data = self.run_config.replay_data(replay['replay_path'])

                        self.process_replay(controller, replay_data, replay['replay_id'], replay['player_id'])

                    except Exception:
                        traceback.print_exc()

                    finally:
                        self.replay_queue.task_done()

    def process_replay(self, controller, replay_data, replay_id, player_id):

        replay_info = controller.replay_info(replay_data)
        map_data = None
        if replay_info.local_map_path: # Special handling for custom maps
            map_data = self.run_config.map_data(replay_info.local_map_path)
        
        for p in replay_info.player_info:
            if p.player_info.player_id == player_id:
                player_race = common_pb.Race.Name(p.player_info.race_actual)
            else:
                enemy_race = common_pb.Race.Name(p.player_info.race_actual)

        global_parser = GlobalParser(player_race, enemy_race)
        spatial_parser = SpatialParser()

        spatial_states_np, global_states_np = [], []
        actions = {}      

        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))
        
        n_states = 0

        print('Parsing', replay_id)

        while True:
            controller.step(FLAGS.step_size)
            obs = controller.observe()

            actions[n_states] = extract_actions(obs)
            spatial_states_np.append(spatial_parser.extract(obs.observation))
            global_states_np.append(global_parser.extract(obs.observation))

            n_states += 1

            if obs.player_result: # Player result obtained means game has ended
                break
        
        spatial_states_np = np.asarray(spatial_states_np)
        global_states_np = np.asarray(global_states_np)

        spatial_states_np = spatial_states_np.reshape((n_states, -1))

        sparse.save_npz(os.path.join(FLAGS.output_path, 'global',
                                    f'{player_id}@{replay_id}.glo'), sparse.csc_matrix(global_states_np)) # Global tensor

        sparse.save_npz(os.path.join(FLAGS.output_path, 'spatial',
                                    f'{player_id}@{replay_id}.spa'), sparse.csc_matrix(spatial_states_np)) # Spatial tensor

        with open(os.path.join(FLAGS.output_path, 'actions', f'{player_id}@{replay_id}.act'), 'w') as f:
            f.write(json.dumps(actions, indent=4))

class ReplayQueue:
    """
    Replay queue

    Unable to inherit directly from multiprocessing.JoinableQueue as it's intended to be used as a method within context
    """
    def __init__(self, queued_replays):
        self.queue = multiprocessing.JoinableQueue(queued_replays)
        self.replays_processed = multiprocessing.Value('i', 0)

    def task_done(self):
        with self.replays_processed.get_lock():
            self.replays_processed.value += 1
        self.queue.task_done()
    
    def put(self, replay):
        self.queue.put(replay)

    def get(self):
        return self.queue.get()

    def join(self):
        self.queue.join()


def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay paths."""

    for replay in replay_list:
        replay_queue.put(replay)

def parse(argv):

    FLAGS.output_path = os.path.join(FLAGS.output_path, f"{FLAGS.player_race}_vs_{FLAGS.enemy_race}")

    for out_folder in ['actions', 'global', 'spatial']:
        path = os.path.join(FLAGS.output_path, out_folder)
        if not os.path.isdir(path):
            os.makedirs(path)

    run_config = run_configs.get()
    try:
        race_vs_race = '_vs_'.join(sorted([FLAGS.player_race, FLAGS.enemy_race]))

        with open(os.path.join(FLAGS.library_path, race_vs_race + '.json')) as f:
            replay_library = json.load(f)

        if FLAGS.map is not None:
            maps = FLAGS.map.split(';')
            replay_library = [replay for replay in replay_library if replay['map'] in maps]

        replay_list = []
        
        for replay in replay_library:
            for player in replay['players']:
                if player['race'] == FLAGS.player_race:
                    replay_id = os.path.basename(replay['path']).replace('.SC2Replay','')

                    if not os.path.isfile(os.path.join(FLAGS.output_path, 'global', f"{player['id']}@{replay_id}.glo.npz")) or \
                        not os.path.isfile(os.path.join(FLAGS.output_path, 'spatial', f"{player['id']}@{replay_id}.spa.npz")) or \
                        not os.path.isfile(os.path.join(FLAGS.output_path, 'actions', f"{player['id']}@{replay_id}.act")):

                        replay_list.append({
                            'replay_path': replay['path'],
                            'replay_id': replay_id,
                            'player_id': player['id']
                        })

        replay_queue = ReplayQueue(FLAGS.n_instance * 10)
        replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                           args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        for i in range(FLAGS.n_instance):
            p = ReplayProcessor(run_config, replay_queue)
            p.daemon = True
            print('Starting thread', i)
            p.start()
            time.sleep(1)   # Stagger startups, otherwise they seem to conflict somehow

        n_replays = len(replay_list)
        n_processed = 0

        pbar = tqdm(total = n_replays, desc='Replays processed')
        while n_processed < n_replays:
            time.sleep(1)
            prev_processed = n_processed
            with replay_queue.replays_processed.get_lock():
                n_processed = replay_queue.replays_processed.value

            pbar.update(n_processed - prev_processed)

        replay_queue.join() # Wait for the queue to empty.

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")

if __name__ == '__main__':
    app.run(parse)