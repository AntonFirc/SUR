from ikrlib import wav16khz2mfcc
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np

class_cnt = 0
gaussian_cnt = 10

gmm_arr = []

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

for speaker_dir in train_path.iterdir():
    if str(speaker_dir).__contains__('.DS_Store'):
        continue

    class_cnt += 1

    speaker_features = []

    for speaker_file in speaker_dir.iterdir():
        if str(speaker_file).endswith('.wav'):
            x, sr = l.load(speaker_file, sr=8000)
            n_fft = int(sr * 0.02)  # window length: 0.02 s
            hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
            mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft)
            # check the dimensions
            speaker_features.append(mfccs)

    features = np.concatenate(speaker_features, axis=1)


print(gmm_arr)