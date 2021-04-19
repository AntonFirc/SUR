import collections
from glob import glob
from multiprocessing.pool import ThreadPool

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

class_cnt = 31
gaussian_cnt = 2

PCA_dim = 60
LDA_dim = 20

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

gmm_arr = {}


class PhotoGenerative:
    wc_cov = None
    class_means = None

    gaussian_cnt = 2
    gmm_ord = None

    dev_path = Path('./dataset/dev')
    train_path = Path('./dataset/train')
    # dev_path = Path('./tmp/faces_new/dev')
    # train_path = Path('./tmp/faces_new/train')

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
    def transform_data(cls, data_src):
        participants = []

        for i in range(class_cnt):
            participant_dir = data_src.joinpath(str(i + 1))
            participants.append(participant_dir)

        feature_list = []
        for participant_dir in participants:
            participant_idx = int(str(participant_dir).split('/').pop())
            photo_features = cls.png2fea(participant_dir)
            feature_list.insert(participant_idx, photo_features)

        photo_feature = feature_list[0]
        img_cnt = len(photo_feature)
        shape = photo_feature[0].shape[0] * photo_feature[0].shape[1]

        reshaped_features = np.array(feature_list).reshape((img_cnt * class_cnt, shape))
        mean_face = np.mean(reshaped_features, axis=0)
        features = reshaped_features - mean_face

        pca = PCA(n_components=60, svd_solver='full', whiten=True)
        pca_fea = pca.fit_transform(features)

        class_labels = []

        for i in range(class_cnt):
            class_labels.append(np.ones(img_cnt) * (i + 1))

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


        # for dev_dir in tqdm(cls.dev_path.iterdir(), 'Eval', len(list(cls.dev_path.iterdir())), unit='speaker'):
        #     if str(dev_dir).__contains__('.DS_Store'):
        #         continue
        #
        #     gt_idx = int(str(dev_dir).split('/').pop())
        #
        #     for person_image in dev_dir.iterdir():
        #         if str(person_image).endswith('.png'):
        #             attempts += 1
        #             pred_class, _ = cls.gmm_eval_person(person_image)
        #             true_accept += 1 if pred_class == gt_idx else 0
        #
        # model_acc = (true_accept / attempts)
        # print('Total accuracy: {0}%'.format(model_acc * 100))

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

# ImageTools.frontalize_dataset(Path('dataset/train'), Path('tmp/faces_new/train'), thread_count=8)
# ImageTools.frontalize_dataset(Path('dataset/dev'), Path('tmp/faces_new/dev'), thread_count=8)
#
# ImageTools.augument_dataset(Path('tmp/faces_new/train'))
#
# exit(1)

pg = PhotoGenerative()

pg.train_gmm()
pg.gmm_eval_model()

# for i in range(3):
#     pg.train_participants()
#     print(pg.wc_cov.sum())
#     print(pg.class_means.sum())
#     pg.evaluate_model()
