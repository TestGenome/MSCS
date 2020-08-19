from absl import app
from absl import flags
from tqdm import tqdm
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

FLAGS = flags.FLAGS
flags.DEFINE_string(name='parsed_replays', default='../parsed_replays/Protoss_vs_Terran',
                    help='Parsed data path')

flags.DEFINE_float(name='test_split', default=0.2,
                    help='Testing split')

def training_split(argv):

    race_vs_race = FLAGS.parsed_replays.split('/')[-1].split('_vs_')

    replays = open(os.path.join(FLAGS.parsed_replays, 'replays.csv')).read().split('\n')[:-1] # -1 to get rid of the final newline

    if race_vs_race[0] == race_vs_race[1]: # If mirrored matchup, prevent the same replay from appearing in both the training and test set
        train, test = train_test_split(np.arange(len(replays)/2), test_size=FLAGS.test_split)
        train = np.concatenate((train * 2, train * 2 + 1)).astype(int)
        test = np.concatenate((test * 2, test * 2 + 1)).astype(int)
    else:
        train, test = train_test_split(np.arange(len(replays)), test_size=FLAGS.test_split)

    with open(os.path.join(FLAGS.parsed_replays, 'train.csv'), 'w') as f:
        for idx in train:
            f.write(f"{replays[idx]}\n")
    
    with open(os.path.join(FLAGS.parsed_replays, 'test.csv'), 'w') as f:
        for idx in test:
            f.write(f"{replays[idx]}\n")

if __name__ == '__main__':
    app.run(training_split)