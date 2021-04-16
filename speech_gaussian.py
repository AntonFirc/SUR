from multiprocessing.pool import ThreadPool
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np
from tqdm import tqdm

class_cnt = 0
gaussian_cnt = 2

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

speakers = []
gmm_arr = []


def process_speaker(speaker_dir):
    speaker_features = []
    speaker_idx = int(str(speaker_dir).split('/').pop())

    print('Processing speaker {0}'.format(speaker_idx))

    for speaker_file in speaker_dir.iterdir():
        if str(speaker_file).endswith('.wav'):
            x, sr = l.load(speaker_file, sr=8000)
            n_fft = int(sr * 0.02)  # window length: 0.02 s
            hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
            mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft)
            speaker_features.append(mfccs)

    features = np.concatenate(speaker_features, axis=1)

    print('Training speaker {0}'.format(speaker_idx))

    gmm = GaussianMixture(n_components=gaussian_cnt, max_iter=10).fit(features)
    gmm_arr.insert(speaker_idx, gmm)

    print(gmm.weights_)
    print(gmm.means_)

    print('Speaker {0} finished!'.format(speaker_idx))

for speaker_dir in train_path.iterdir():
    if str(speaker_dir).__contains__('.DS_Store'):
        continue

    class_cnt += 1
    speaker_features = []
    speaker_idx = int(str(speaker_dir).split('/').pop())

    speakers.append(speaker_dir)

with ThreadPool(1) as pool:
    list(
        tqdm(
            pool.imap(
                process_speaker,
                speakers
            ),
            'Process',
            len(speakers),
            unit="speakers"
        )
    )

print(gmm_arr)
