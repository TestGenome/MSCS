import os
import json
import glob
from absl import app
from absl import flags
from tqdm import tqdm
from itertools import chain

from google.protobuf.json_format import Parse

from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='replays_paths', default='./;',
                    help='Paths for replays, split by ;')
flags.DEFINE_string(name='save_path', default='../replay_library',
                    help='Path for saving results')

flags.DEFINE_integer(name='min_duration', default=8000,
                     help='Min duration')
flags.DEFINE_integer(name='max_duration', default=80000,
                     help='Max duration')
flags.DEFINE_integer(name='min_apm', default=10,
                     help='Min APM')
flags.DEFINE_integer(name='min_mmr', default=1000,
                     help='Min MMR')

def get_replay_info(run_config, controller, replay_path):
    replay_data = run_config.replay_data(replay_path)

    try:
        info = controller.replay_info(replay_data) # Get high level replay info
    except Exception:
        print('Unable to get replay data for', replay_path)
        return None

    if info.HasField("error"):
        return None
    if info.base_build != controller.ping().base_build: # Make sure replay has same version
        return None
    if info.game_duration_loops < FLAGS.min_duration:
        return None
    if info.game_duration_loops > FLAGS.max_duration:
        return None
    if len(info.player_info) != 2:
        return None

    replay_info = {
        'path': replay_path,
        'map': info.map_name,
        'duration_seconds': info.game_duration_seconds,
        'duration_frames': info.game_duration_loops,
        'players': []
    }

    for p in info.player_info:
        if p.player_apm < FLAGS.min_apm or p.player_mmr < FLAGS.min_mmr:
            return None
        if p.player_result.result not in {1, 2}: # 1 is victory, 2 is defeat
            return None

        replay_info['players'].append({
            'id': p.player_info.player_id,
            'race': common_pb.Race.Name(p.player_info.race_actual),
            'result': 2 - p.player_result.result,
            'apm': p.player_apm,
            'mmr': p.player_mmr
        })

    return replay_info

def preprocess(argv):

    if not os.path.isdir(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    run_config = run_configs.get()
    replay_list = sorted(chain(*[run_config.replay_paths(path)
                            for path in FLAGS.replays_paths.split(';')
                                if len(path.strip()) > 0]))

    result = {}
    stats = {}
    totals = {}

    min_frames = 999999999

    try:
        with run_config.start() as controller:

            pbar = tqdm(total=len(replay_list), desc='Replays processed')

            for replay_path in replay_list:
                replay_info = get_replay_info(run_config, controller, replay_path)

                if replay_info is not None:

                    races = '_vs_'.join(sorted(player['race'] for player in replay_info['players']))
                    if races not in result:
                        result[races] = []
                        totals[races] = 0

                    min_frames = min(min_frames, replay_info['duration_frames'])

                    totals[races] += 1
                    result[races].append(replay_info) # Save based on race vs race

                    if replay_info['map'] not in stats:
                        stats[replay_info['map']] = {}

                    if races not in stats[replay_info['map']]:
                        stats[replay_info['map']][races] = 0

                    stats[replay_info['map']][races] += 1
                
                pbar.update()

        for k, v in result.items():
            with open(os.path.join(FLAGS.save_path, k+'.json'), 'w') as f:
                f.write(json.dumps(v, indent=4))

        with open(os.path.join(FLAGS.save_path, 'stats.txt'), 'w') as f:
            for k, v in totals.items():
                f.write(f"{k} {v}\n")

            f.write("\n")

            for k, v in stats.items():
                f.write(f"{k}\n")
                for r, b in v.items():
                    f.write(f"    {r} {b}\n")

                f.write("\n")

        print(min_frames)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")

if __name__ == '__main__':
    app.run(preprocess)