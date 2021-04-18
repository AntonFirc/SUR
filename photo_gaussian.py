import collections
from glob import glob
from multiprocessing.pool import ThreadPool
from image_tools import ImageTools

from matplotlib import pyplot as plt

from imread import imread
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np
from tqdm import tqdm
import subprocess

class_cnt = 31
gaussian_cnt = 2

PCA_dim = 60
LDA_dim = 20

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

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


def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = []
    for f in glob(str(dir_name) + '/*.png'):
        features.append(calc_feature(f))
    return features


def calc_feature(file):
    return imread(file, True).astype(np.float64)


def train_participants():
    for i in range(class_cnt):
        participant_dir = train_path.joinpath(str(i + 1))
        participants.append(participant_dir)

    # feature_arr = {}
    # for participant_dir in participants:
    #     participant_idx = int(str(participant_dir).split('/').pop())
    #     photo_features = png2fea(participant_dir)
    #     feature_arr[participant_idx] = photo_features

    feature_list = []
    for participant_dir in participants:
        participant_idx = int(str(participant_dir).split('/').pop())
        photo_features = png2fea(participant_dir)
        feature_list.insert(participant_idx, photo_features)

    # features = np.concatenate(photo_features, axis=0)

    reshaped_features = np.array(feature_list).reshape((6*31, 6400))
    mean_face = np.mean(reshaped_features, axis=0)
    features = reshaped_features - mean_face

    # Display mean face
    # plt.imshow(mean_face.reshape((80, 80)), interpolation='nearest')
    # plt.show()

    u, s, _ = np.linalg.svd(features)
    reshaped_u = u[:, :PCA_dim]
    train_pca = reshaped_u.T @ features

    # Display PCAs
    # for pca in train_pca:
    #     plt.imshow(pca.reshape((80, 80)), interpolation='nearest')
    #     plt.show()

    # gmm = GaussianMixture(n_components=gaussian_cnt, max_iter=200).fit(features)
    # # gmm_arr[participant_idx] = gmm

    # LDA
    wc_cov = np.zeros((PCA_dim, PCA_dim))
    for i in range(class_cnt):
        wc_cov += wc_cov + np.cov(train_pca())


def eval_participant(recording_path):
    scores = []

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



train_participants()

gmm_ord = collections.OrderedDict(sorted(gmm_arr.items()))

evaluate_model()
