import os.path
import subprocess
from multiprocessing.pool import ThreadPool
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np
from tqdm import tqdm
import collections
from audio_tools import AudioTools


class SpeechGaussian:
    speakers = []
    speakers_iter = {}
    gmm_arr = {}
    gmm_ord = {}

    class_cnt = 31
    gaussian_cnt = 2

    dev_orig_path = Path('./dataset/dev')
    train_orig_path = Path('./dataset/train')

    dev_path = Path('./tmp/dev')
    train_path = Path('./tmp/train')

    @classmethod
    def waw_2_mfcc(cls, waw_path):
        x, sr = l.load(waw_path, sr=16000)
        n_fft = int(sr * 0.02)  # window length: 0.02 s
        hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
        mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft).T
        mfcc_delta = np.concatenate((np.zeros((1, 24)), np.diff(mfccs, axis=0)), axis=0)
        return np.concatenate((mfccs, mfcc_delta), axis=1)

    @classmethod
    def train_speaker(cls, speaker_dir):
        speaker_features = []
        speaker_idx = int(str(speaker_dir).split('/').pop())

        for speaker_file in speaker_dir.iterdir():
            if str(speaker_file).endswith('.wav'):
                speaker_features.append(cls.waw_2_mfcc(speaker_file))

        features = np.concatenate(speaker_features, axis=0)

        gmm = GaussianMixture(n_components=cls.gaussian_cnt, max_iter=2000).fit(features)
        cls.gmm_arr[speaker_idx] = gmm

    @classmethod
    def eval_speaker(cls, recording_path):
        scores = []

        try:
            recording_mfcc = cls.waw_2_mfcc(recording_path)
        except ValueError:
            return -1

        for key in cls.gmm_ord:
            scores.append(sum(cls.gmm_ord[key].score_samples(recording_mfcc)))

        np_s = np.array(scores)
        return np_s.argmax() + 1, np_s

    @classmethod
    def evaluate_model(cls):
        attempts = 0
        true_accept = 0

        for dev_dir in tqdm(cls.dev_path.iterdir(), 'Eval', len(list(cls.dev_path.iterdir())), unit='speaker'):
            if str(dev_dir).__contains__('.DS_Store'):
                continue

            gt_idx = int(str(dev_dir).split('/').pop())

            for speaker_file in dev_dir.iterdir():
                if str(speaker_file).endswith('.wav'):
                    attempts += 1
                    pred_class, _ = cls.eval_speaker(speaker_file)
                    true_accept += 1 if pred_class == gt_idx else 0

        model_acc = (true_accept / attempts)
        print('Total accuracy: {0}%'.format(model_acc * 100))

    @classmethod
    def label_data(cls, eval_dir):
        result_file = open("speech_gaussian.txt", "w")

        for eval_file in tqdm(eval_dir.iterdir(), 'Label data', len(list(eval_dir.iterdir())), unit='files'):
            if str(eval_file).endswith('.wav'):
                pred_class, probs = cls.eval_speaker(eval_file)
                res_line = '{0} {1} {2}\n'.format(os.path.basename(eval_file).replace('.wav', ''), pred_class,
                                                  ' '.join(str(x) for x in probs))
                result_file.write(res_line)

        result_file.close()

    @classmethod
    def train_basic(cls):
        for i in range(cls.class_cnt):
            speaker_dir = cls.train_path.joinpath(str(i + 1))
            cls.speakers.append(speaker_dir)

        with ThreadPool(8) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.train_speaker,
                        cls.speakers
                    ),
                    'Train',
                    len(cls.speakers),
                    unit="speaker"
                )
            )

        cls.gmm_ord = collections.OrderedDict(sorted(cls.gmm_arr.items()))


# for speaker_dir in train_path.iterdir():
#     if str(speaker_dir).__contains__('.DS_Store'):
#         continue
#
#     class_cnt += 1
#     speaker_features = []
#     speaker_idx = int(str(speaker_dir).split('/').pop())
#
#     speakers.append(speaker_dir)

rm_tmp = [
    'rm',
    '-rf',
    './tmp',
]
subprocess.call(rm_tmp)

sg = SpeechGaussian()

AudioTools.sox_prepare_dataset(sg.train_orig_path, sg.train_path)
AudioTools.sox_prepare_dataset(sg.dev_orig_path, sg.dev_path)

AudioTools.sox_augument_dataset(sg.train_path, [0.9, 0.95, 1.05, 1.1])

sg.train_basic()
sg.evaluate_model()
sg.label_data(Path('dataset/eval'))
