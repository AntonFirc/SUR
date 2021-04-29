import os.path
from multiprocessing.pool import ThreadPool
from sklearn.mixture import GaussianMixture
import librosa as l
from pathlib import Path
import numpy as np
from tqdm import tqdm
import collections
import warnings

# ignores librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SpeechGaussian:
    speakers = []
    speakers_iter = {}
    gmm_arr = {}
    gmm_ord = {}

    class_cnt = 31
    gaussian_cnt = 2

    dev_orig_path = Path('./dataset/dev')
    eval_orig_path = Path('./dataset/eval')
    train_orig_path = Path('./dataset/train_big')

    dev_path = Path('./temp/dev')
    eval_path = Path('./temp/eval')
    train_path = Path('./temp/train_big')

    @classmethod
    def waw_2_mfcc(cls, waw_path):
        x, sr = l.load(waw_path, sr=16000)
        n_fft = int(sr * 0.02)  # window length: 0.02 s
        hop_length = n_fft // 2  # usually one specifies the hop length as a fraction of the window length
        mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=24, hop_length=hop_length, n_fft=n_fft).T
        mfcc_delta = np.concatenate((np.zeros((1, 24)), np.diff(mfccs, axis=0)), axis=0)
        return np.concatenate((mfccs, mfcc_delta), axis=1)

    @classmethod
    def gmm_train_speaker(cls, speaker_dir):
        speaker_features = []
        speaker_idx = int(str(speaker_dir).split('/').pop())

        for speaker_file in speaker_dir.iterdir():
            if str(speaker_file).endswith('.wav'):
                speaker_features.append(cls.waw_2_mfcc(speaker_file))

        features = np.concatenate(speaker_features, axis=0)

        gmm = GaussianMixture(n_components=cls.gaussian_cnt, max_iter=2000).fit(features)
        cls.gmm_arr[speaker_idx] = gmm

    @classmethod
    def gmm_eval_speaker(cls, recording_path):
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
    def gmm_evaluate_model(cls):
        attempts = 0
        true_accept = 0

        for dev_dir in tqdm(cls.dev_path.iterdir(), 'Eval', len(list(cls.dev_path.iterdir())), unit='speaker'):
            if str(dev_dir).__contains__('.DS_Store'):
                continue

            gt_idx = int(str(dev_dir).split('/').pop())

            for speaker_file in dev_dir.iterdir():
                if str(speaker_file).endswith('.wav'):
                    attempts += 1
                    try:
                        pred_class, _ = cls.gmm_eval_speaker(speaker_file)
                    except TypeError:
                        print(str(speaker_file))
                        continue
                    true_accept += 1 if pred_class == gt_idx else 0

        model_acc = (true_accept / attempts)
        print('Total accuracy: {0}%'.format(model_acc * 100))

    @classmethod
    def gmm_label_data(cls, eval_dir):
        result_file = open("speech_gaussian.txt", "w")

        for eval_file in tqdm(eval_dir.iterdir(), 'Label data', len(list(eval_dir.iterdir())), unit='files'):
            if str(eval_file).endswith('.wav'):
                try:
                    pred_class, probs = cls.gmm_eval_speaker(eval_file)
                except TypeError:
                    print(str(eval_file))
                    continue
                res_line = '{0} {1} {2}\n'.format(os.path.basename(eval_file).replace('.wav', ''), pred_class,
                                                  ' '.join(str(x) for x in probs))
                result_file.write(res_line)

        result_file.close()

    @classmethod
    def train_gmm(cls):
        print(cls)
        for i in range(cls.class_cnt):
            speaker_dir = cls.train_path.joinpath(str(i + 1))
            cls.speakers.append(speaker_dir)

        with ThreadPool(8) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.gmm_train_speaker,
                        cls.speakers
                    ),
                    'Train',
                    len(cls.speakers),
                    unit="speaker"
                )
            )

        cls.gmm_ord = collections.OrderedDict(sorted(cls.gmm_arr.items()))
