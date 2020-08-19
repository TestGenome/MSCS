from absl import app
from absl import flags
from tqdm import tqdm
import os
import glob
import numpy as np
from scipy import sparse
import csv

FLAGS = flags.FLAGS
flags.DEFINE_string(name='parsed_replays', default='../parsed_replays/Protoss_vs_Terran',
                    help='Parsed data path')

def postprocess(argv):

    feature_names = np.loadtxt(os.path.join(FLAGS.parsed_replays, 'features.csv'), delimiter=',', usecols=(1), unpack=True, dtype='str')

    max_cols = np.zeros(len(feature_names)) 
    max_replay = ['none'] * len(feature_names)

    # global_files = glob.glob(os.path.join(FLAGS.parsed_replays, 'global', '*.glo.npz'))

    # pbar = tqdm(total=len(global_files), desc='Replays processed')
    replays = np.loadtxt(os.path.join(FLAGS.parsed_replays, 'replays.csv'), delimiter=',', usecols=(0), unpack=True, dtype='str')

    for replay in tqdm(replays, desc='Scanning'):
        global_np = np.asarray(sparse.load_npz(os.path.join(FLAGS.parsed_replays, 'global', f"{replay}.glo.npz")).todense())

        max_counts = np.max(global_np, axis=0)

        for i in range(len(feature_names)):
            if max_counts[i] > max_cols[i]:
                max_cols[i] = max_counts[i]
                max_replay[i] = replay

    with open(os.path.join(FLAGS.parsed_replays, 'maximums.csv'), 'w') as f:
        for i, name in enumerate(feature_names):
            f.write(f"{name},{max_cols[i]},{max_replay[i]}\n")
            
    max_cols[max_cols == 0] = 1
    max_cols = max_cols[np.newaxis, :]

    out_path = os.path.join(FLAGS.parsed_replays, 'global_normal')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    for replay in tqdm(replays, desc='Normalising'):
        global_np = np.asarray(sparse.load_npz(os.path.join(FLAGS.parsed_replays, 'global', f"{replay}.glo.npz")).todense())
        global_np /= max_cols
        sparse.save_npz(os.path.join(out_path, f"{replay}.glo"), sparse.csc_matrix(global_np)) # Global tensor

if __name__ == '__main__':
    app.run(postprocess)