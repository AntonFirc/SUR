from multiprocessing.pool import ThreadPool
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np
from tqdm import tqdm
import collections

class_cnt = 31
gaussian_cnt = 10

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

speakers = []
gmm_arr = {}


def train_speaker(speaker_dir):
    speaker_features = []
    speaker_idx = int(str(speaker_dir).split('/').pop())

    for speaker_file in speaker_dir.iterdir():
        if str(speaker_file).endswith('.wav'):
            x, sr = l.load(speaker_file, sr=8000)
            n_fft = int(sr * 0.02)  # window length: 0.02 s
            hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
            mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft).T
            speaker_features.append(mfccs)

    features = np.concatenate(speaker_features, axis=0)

    gmm = GaussianMixture(n_components=gaussian_cnt, max_iter=10).fit(features)
    gmm_arr[speaker_idx] = gmm


def eval_speaker(recording_path):
    scores = []

    x, sr = l.load(recording_path, sr=8000)
    n_fft = int(sr * 0.02)  # window length: 0.02 s
    hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
    mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft).T

    for key in gmm_ord:
        scores.append(sum(gmm_ord[key].score_samples(mfccs)))

    np_s = np.array(scores)
    return np_s.argmax() + 1


def evaluate_model():
    attempts = 0
    true_accept = 0

    for dev_dir in train_path.iterdir():
        if str(dev_dir).__contains__('.DS_Store'):
            continue

        gt_idx = int(str(dev_dir).split('/').pop())

        for speaker_file in dev_dir.iterdir():
            if str(speaker_file).endswith('.wav'):
                attempts += 1
                pred_class = eval_speaker(speaker_file)
                #print('{0} / {1}'.format(pred_class, gt_idx))
                true_accept += 1 if pred_class == gt_idx else 0

    model_acc = (true_accept / attempts)
    print('Total accuracy: {0}%'.format(model_acc * 100))


for i in range(class_cnt):
    speaker_dir = train_path.joinpath(str(i + 1))
    speakers.append(speaker_dir)

# for speaker_dir in train_path.iterdir():
#     if str(speaker_dir).__contains__('.DS_Store'):
#         continue
#
#     class_cnt += 1
#     speaker_features = []
#     speaker_idx = int(str(speaker_dir).split('/').pop())
#
#     speakers.append(speaker_dir)

with ThreadPool(8) as pool:
    list(
        tqdm(
            pool.imap(
                train_speaker,
                speakers
            ),
            'Process',
            len(speakers),
            unit="speakers"
        )
    )

gmm_ord = collections.OrderedDict(sorted(gmm_arr.items()))

evaluate_model()

# weights = []
# means = []
# covs = []
#
# for i in range(len(gmm_arr)):
#     weights.append(gmm_arr[i].weights_)
#     means.append(gmm_arr[i].means_)
#     covs.append(gmm_arr[i].covariances_)
#
# np.save('weights', np.array(weights))
# np.save('covs', np.array(covs))
# np.save('means', np.array(means))

# def load_gmm():
#     weights = np.load('weights.npy')
#     covs = np.load('covs.npy')
#     means = np.load('means.npy')
#
#     assert len(weights) == len(covs) == len(means)
#
#     nb_components = len(weights)
#
#     for i in range(nb_components):
#         gmm = GaussianMixture(n_components=nb_components, max_iter=2000)
#         gmm.means_ = means[i]
#         gmm.covariances_ = covs[i]
#         gmm.weights_ = weights[i]
#         gmm_arr.insert(i + 1, gmm)
