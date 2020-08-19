from extract_global import GlobalParser
from extract_spatial import SpatialParser

import os
import sys
import json
from absl import app
from absl import flags

from scipy import sparse
import numpy as np

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

def is_valid_replay(replay, player):

    if player['race'] != FLAGS.player_race:
        return False, None

    replay_id = os.path.basename(replay['path']).replace('.SC2Replay','')

    if not os.path.isfile(os.path.join(FLAGS.output_path, 'global', f"{player['id']}@{replay_id}.glo.npz")):
        return False, -1
    if not os.path.isfile(os.path.join(FLAGS.output_path, 'spatial', f"{player['id']}@{replay_id}.spa.npz")):
        return False, -2
    if not os.path.isfile(os.path.join(FLAGS.output_path, 'actions', f"{player['id']}@{replay_id}.act")):
        return False, -3
        
    glo = np.asarray(sparse.load_npz(os.path.join(FLAGS.output_path, 'global', f"{player['id']}@{replay_id}.glo.npz")).todense())

    if (replay['duration_frames'] - glo[-1,0]) > (glo[1,0] * 10): # If final frame cut off too early, give 10x leeway
        return False, glo.shape[0]
        
    return True, glo.shape[0]

def finalise(argv):

    FLAGS.output_path = os.path.join(FLAGS.output_path, f"{FLAGS.player_race}_vs_{FLAGS.enemy_race}")

    with open(os.path.join(FLAGS.output_path, 'features.csv'), 'w') as f:
        global_parser = GlobalParser(FLAGS.player_race, FLAGS.enemy_race)
        features = global_parser.get_feature_list()

        for i, feature in enumerate(features):
            f.write(f"{i:3},{feature}\n")

    with open(os.path.join(FLAGS.output_path, 'spa_embedding.csv'), 'w') as f:
        spatial_parser = SpatialParser()

        for name, category, scale in spatial_parser.get_scale():
            f.write(f"{name},{category},{scale}\n")

    # with open(os.path.join(FLAGS.output_path, 'minimap_units.csv'), 'w') as f:
    #     pass

    with open(os.path.join(FLAGS.library_path, '_vs_'.join(sorted([FLAGS.player_race, FLAGS.enemy_race])) + '.json')) as f:
        replay_library = json.load(f) 

    with open(os.path.join(FLAGS.output_path, 'replays.csv'), 'w') as f:
        
        for replay in tqdm(replay_library, desc = 'Finalising'):
            for player in replay['players']:
                valid, steps = is_valid_replay(replay, player)

                if valid:
                    f.write(f"{player['id']}@{os.path.basename(replay['path']).split('.')[0]},{player['result']},{steps}\n")
                    
                    
if __name__ == '__main__':
    app.run(finalise)