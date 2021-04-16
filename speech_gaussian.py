from multiprocessing.pool import ThreadPool
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np
from tqdm import tqdm

do_train = True

class_cnt = 0
gaussian_cnt = 2

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

speakers = []
gmm_arr = []


def load_gmm():
    weights = np.load('weights.npy')
    covs = np.load('covs.npy')
    means = np.load('means.npy')

    assert len(weights) == len(covs) == len(means)

    nb_components = len(weights)

    for i in range(nb_components):
        gmm = GaussianMixture(n_components=nb_components, max_iter=10)
        gmm.means_ = means[i]
        gmm.covariances_ = covs[i]
        gmm.weights_ = weights[i]
        gmm_arr.insert(i + 1, gmm)


def train_speaker(speaker_dir):
    speaker_features = []
    speaker_idx = int(str(speaker_dir).split('/').pop())

    #print('Processing speaker {0}'.format(speaker_idx))

    for speaker_file in speaker_dir.iterdir():
        if str(speaker_file).endswith('.wav'):
            x, sr = l.load(speaker_file, sr=8000)
            n_fft = int(sr * 0.02)  # window length: 0.02 s
            hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
            mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft).T
            speaker_features.append(mfccs)

    features = np.concatenate(speaker_features, axis=0)

    #print('Training speaker {0}'.format(speaker_idx))

    gmm = GaussianMixture(n_components=gaussian_cnt, max_iter=10).fit(features)
    gmm_arr.insert(speaker_idx, gmm)

    #print('Speaker {0} finished!'.format(speaker_idx))


def speaker_score():
    scores = []

    x, sr = l.load('./dataset/dev/1/f401_04_f13_i0_0.wav', sr=8000)
    n_fft = int(sr * 0.02)  # window length: 0.02 s
    hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
    mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft).T

    for gmm in gmm_arr:
        scores.append(sum(gmm.score_samples(mfccs)))

    np_s = np.array(scores)
    print(np_s.argmax())


if not do_train:
    load_gmm()
    speaker_score()
else:
    for speaker_dir in train_path.iterdir():
        if str(speaker_dir).__contains__('.DS_Store'):
            continue

        class_cnt += 1
        speaker_features = []
        speaker_idx = int(str(speaker_dir).split('/').pop())

        speakers.append(speaker_dir)

    with ThreadPool(4) as pool:
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

    speaker_score()
