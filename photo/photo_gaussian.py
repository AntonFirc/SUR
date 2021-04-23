import collections
from glob import glob
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot
import scipy.sparse.linalg

from image_tools import ImageTools

from matplotlib import pyplot as plt

from imread import imread
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pathlib import Path
import numpy as np
from tqdm import tqdm
import subprocess
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

class_cnt = 15

PCA_dim = 60
LDA_dim = 30

dev_path = Path('../dataset/dev')
train_path = Path('../dataset/train')

gmm_arr = {}


class PhotoGenerative:
    wc_cov = None
    class_means = None

    gaussian_cnt = 15
    gmm_ord = None

    kernels = []

    # dev_path = Path('./dataset/dev')
    # train_path = Path('./dataset/train')
    # dev_path = Path('./tmp/faces_new/dev')
    # train_path = Path('./tmp/faces_new/train')
    dev_path = Path('../dataset/dev2')
    train_path = Path('../dataset/yale')

    @classmethod
    def png2fea(cls, dir_name):
        """
        Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
        and values and 2D numpy arrays with corresponding grayscale images
        """
        features = []
        for f in glob(str(dir_name) + '/*.png'):
            features.append(cls.calc_feature(f))
        return features

    @classmethod
    def calc_feature(cls, file):
        return imread(file, True).astype(np.float64)

    @classmethod
    def compute_features(cls, image):
        feats = []

        for k, kernel in enumerate(cls.kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats.append(filtered)

        feats = np.stack(feats).flatten()
        return feats

    @classmethod
    def transform_gabor(cls, data_src):
        for theta in range(8):
            theta = theta / 4. * np.pi
            for sigma in (1, 5):
                for frequency in (0.05, 0.4):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    cls.kernels.append(kernel)

        participants = []
        class_labels = []
        feature_list = []

        for i in range(class_cnt):
            participant_dir = data_src.joinpath(str(i + 1))
            participants.append(participant_dir)

        for participant_dir in tqdm(participants, 'Gabor demodulation', len(participants), unit='person'):
            participant_idx = int(str(participant_dir).split('/').pop())
            photo_features = []
            for photo in participant_dir.iterdir():
                # if not str(photo).endswith('.png'):
                #     continue
                img = cls.calc_feature(photo)
                photo_features.append(cls.compute_features(img))
            class_labels.append(np.ones(len(photo_features)) * participant_idx)
            feature_list.insert(participant_idx, photo_features)

        reshaped_features = np.concatenate(feature_list)
        target_labels = np.concatenate(class_labels)

        pca = PCA(n_components=PCA_dim, svd_solver='full', whiten=True)
        pca_fea = pca.fit_transform(reshaped_features)

        lda = LDA(n_components=LDA_dim)
        lda_data = lda.fit_transform(pca_fea, target_labels)

        return lda_data, target_labels


    @classmethod
    def transform_data(cls, data_src):
        participants = []

        for i in range(class_cnt):
            participant_dir = data_src.joinpath(str(i + 1))
            participants.append(participant_dir)

        feature_list = []
        class_labels = []
        for participant_dir in participants:
            participant_idx = int(str(participant_dir).split('/').pop())
            photo_features = cls.png2fea(participant_dir)
            class_labels.append(np.ones(len(photo_features)) * participant_idx)
            photo_features_reshaped = np.array(photo_features).reshape(len(photo_features), photo_features[0].shape[0] *
                                                                       photo_features[0].shape[1])
            feature_list.insert(participant_idx, photo_features_reshaped)

        reshaped_features = np.concatenate(feature_list)

        pca = PCA(n_components=PCA_dim, svd_solver='full', whiten=True)
        pca_fea = pca.fit_transform(reshaped_features)

        target_labels = np.concatenate(class_labels)

        lda = LDA(n_components=LDA_dim)
        lda_data = lda.fit_transform(pca_fea, target_labels)
        return lda_data, target_labels

    @classmethod
    def train_participants(cls):
        lda_train, _ = cls.transform_data(cls.train_path)

        class_means = []
        wc_cov = np.zeros((LDA_dim, LDA_dim))
        for i in range(class_cnt):
            lda_fea_class = lda_train[i * 6:i * 6 + 5, :].T
            wc_cov += wc_cov + np.cov(lda_fea_class)
            class_means.insert(i, np.mean(lda_fea_class, axis=1))

        class_means = np.array(class_means)

        cls.wc_cov = wc_cov / class_cnt
        cls.class_means = class_means

    @classmethod
    def train_gmm(cls):
        lda_train, _ = cls.transform_data(cls.train_path)

        cls.gmm_ord = OrderedDict()

        for i in range(class_cnt):
            lda_fea_class = lda_train[i * 6:i * 6 + 5, :]
            cls.gmm_ord[i] = GaussianMixture(n_components=cls.gaussian_cnt, max_iter=2000).fit(lda_fea_class)

    @classmethod
    def gmm_eval_person(cls, lda_data):
        scores = []

        for key in cls.gmm_ord:
            scores.append(sum(cls.gmm_ord[key].score_samples(lda_data.T)))

        np_s = np.array(scores)
        return np_s.argmax() + 1, np_s

    @classmethod
    def gmm_eval_model(cls):
        attempts = 0
        results = []

        lda_eval, target_labels = cls.transform_data(cls.dev_path)

        for person in tqdm(lda_eval, 'Eval', len(lda_eval), unit='image'):
            attempts += 1
            pred_class, _ = cls.gmm_eval_person(person.reshape(-1, 1))
            results.append(pred_class)

        res_np = np.stack(results)
        err = np.count_nonzero(res_np - target_labels)
        acc = 1 - (err / len(target_labels))

        print('Total accuracy: {0}%'.format(acc * 100))

    @classmethod
    def mlp_train(cls):
        # lda_data, labels = cls.transform_gabor(cls.train_path)
        lda_data, labels = cls.transform_gabor(cls.train_path)

        # clf = MLPClassifier(random_state=1, activation='logistic', solver='sgd', learning_rate='adaptive',
        #                     early_stopping=True, max_iter=1000).fit(lda_data, labels)
        #clf = MLPClassifier(max_iter=1000, early_stopping=True, solver='sgd', learning_rate='invscaling', alpha=0.5, hidden_layer_sizes=(100, 100,)).fit(lda_data, labels)
        clf = MLPClassifier(early_stopping=True, max_iter=500).fit(lda_data, labels)

        eval_data, eval_labels = cls.transform_data(cls.train_path)

        log_probs = clf.predict_log_proba(eval_data)

        class_idx = np.argmax(log_probs, axis=1) + 1

        err = np.count_nonzero(class_idx - eval_labels)
        acc = 1 - (err / len(eval_labels))

        print('Total accuracy: {0}%'.format(acc * 100))

    @classmethod
    def classify_data(cls, lda_data):
        res_arr = []

        for i in range(class_cnt):
            w_k = cls.class_means[i] @ np.linalg.inv(cls.wc_cov)
            w_0_k = -0.5 * cls.class_means[i] @ np.linalg.inv(cls.wc_cov) @ cls.class_means[i].T

            first = w_k @ lda_data.T
            second = first + w_0_k

            res_arr.append(second)

        res_np = np.stack(res_arr, axis=0)

        res_np = np.log(res_np)

        win_idx = res_np.argmax(axis=0)

        return win_idx + 1, res_np

    @classmethod
    def label_data(cls):
        pass

    @classmethod
    def evaluate_model(cls):
        lda_eval, target_labels = cls.transform_data(cls.train_path)

        class_idx, _ = cls.classify_data(lda_eval)

        err = np.count_nonzero(class_idx - target_labels)
        acc = 1 - (err / len(target_labels))

        print('Total accuracy: {0}%'.format(acc * 100))


# rm_tmp = [
#     'rm',
#     '-rf',
#     './tmp/faces',
# ]
# subprocess.call(rm_tmp)
#
# ImageTools.frontalize_dataset(Path('dataset/train'), Path('tmp/faces_new/train'), thread_count=8)
# ImageTools.frontalize_dataset(Path('dataset/dev'), Path('tmp/faces_new/dev'), thread_count=8)
#
# ImageTools.augument_dataset(Path('tmp/faces_new/train'))
#
# exit(1)

pg = PhotoGenerative()

pg.mlp_train()

#
# pg.train_gmm()
# pg.gmm_eval_model()

# for i in range(3):
#     pg.train_participants()
#     print(pg.wc_cov.sum())
#     print(pg.class_means.sum())
#     pg.evaluate_model()
