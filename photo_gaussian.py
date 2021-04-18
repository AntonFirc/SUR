import collections
from multiprocessing.pool import ThreadPool
from image_tools import ImageTools
from imread import imread
from sklearn.mixture import GaussianMixture
from pathlib import Path
import numpy as np
from tqdm import tqdm
import subprocess

import png

class_cnt = 31
gaussian_cnt = 2

dev_path = Path('./tmp/faces/dev')
train_path = Path('./tmp/faces/train')

participants = []
gmm_arr = {}

# rm_tmp = [
#     'rm',
#     '-rf',
#     './tmp/faces',
# ]
# subprocess.call(rm_tmp)
#
# ImageTools.frontalize_dataset(Path('dataset/train'), Path('tmp/faces/train'), thread_count=8)
# ImageTools.frontalize_dataset(Path('dataset/dev'), Path('tmp/faces/dev'), thread_count=8)
#
# ImageTools.augument_dataset(Path('tmp/faces/train'))


def train_participant(photo_dir):
    participant_idx = int(str(photo_dir).split('/').pop())
    photo_features = png.png2fea(photo_dir)

    features = np.concatenate(photo_features, axis=0)

    gmm = GaussianMixture(n_components=gaussian_cnt, max_iter=200).fit(features)
    gmm_arr[participant_idx] = gmm


def eval_participant(recording_path):
    scores = []
    # recording_path = './dataset/train/28/m429_01_f13_i0_0.png'

    f = recording_path
    mfccs = imread(f, True).astype(np.float64)

    for key in gmm_ord:
        scores.append(sum(gmm_ord[key].score_samples(mfccs)))

    np_s = np.array(scores)
    # print(np_s.argmax() + 1)
    return np_s.argmax() + 1


def evaluate_model():
    attempts = 0
    true_accept = 0

    for dev_dir in dev_path.iterdir():
        if str(dev_dir).__contains__('.DS_Store'):
            continue

        gt_idx = int(str(dev_dir).split('/').pop())

        for speaker_file in dev_dir.iterdir():
            if str(speaker_file).endswith('.png'):
                attempts += 1
                pred_class = eval_participant(speaker_file)
                # print('{0} / {1}'.format(pred_class, gt_idx))
                true_accept += 1 if pred_class == gt_idx else 0

    model_acc = (true_accept / attempts)
    print('Total accuracy: {0}%'.format(model_acc * 100))


for i in range(class_cnt):
    participant_dir = train_path.joinpath(str(i + 1))
    participants.append(participant_dir)

with ThreadPool(1) as pool:
    list(
        tqdm(
            pool.imap(
                train_participant,
                participants
            ),
            'Train',
            len(participants),
            unit="participant"
        )
    )

gmm_ord = collections.OrderedDict(sorted(gmm_arr.items()))

evaluate_model()
# print(gmm_arr)
