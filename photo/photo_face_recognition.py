import face_recognition as fr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.special
import collections
import os


class PhotoFaceRecognition:
    TRAIN_DIR = Path('./dataset_clean/train')
    # DEV_DIR = Path('./dataset_clean/train')
    DEV_DIR = Path('./dataset/dev2_full')
    EVAL_DIR = Path('./dataset/eval')

    known_faces = []
    known_classes = []

    @classmethod
    def load_image(cls, file_path):
        img = fr.load_image_file(file_path)
        locations = fr.face_locations(img, number_of_times_to_upsample=3, model="cnn")
        return np.array(fr.face_encodings(img, known_face_locations=locations, num_jitters=10, model="large"))

    @classmethod
    def train_model(cls):
        for person_folder in tqdm(cls.TRAIN_DIR.iterdir(), 'Train', len(list(cls.TRAIN_DIR.iterdir())), unit='person'):

            if str(person_folder).__contains__('.DS_Store'):
                continue

            person_idx = int(str(person_folder).split('/').pop())
            person_faces = []

            for person_file in person_folder.iterdir():
                if not str(person_file).endswith('.png'):
                    continue

                encoding = cls.load_image(person_file)

                if encoding.shape[0] != 0:
                    person_faces.append(encoding[0])

            cls.known_classes.append(person_idx)
            cls.known_faces.append(np.mean(person_faces, axis=0))

    @classmethod
    def eval_model(cls):
        attempts = 0
        matches = 0

        for dev_folder in tqdm(cls.DEV_DIR.iterdir(), 'Eval', len(list(cls.DEV_DIR.iterdir())), unit='person'):
            if str(dev_folder).__contains__('.DS_Store'):
                continue

            person_idx = int(str(dev_folder).split('/').pop())

            for dev_file in dev_folder.iterdir():
                if not str(dev_file).endswith('.png'):
                    continue

                encoding = cls.load_image(dev_file)

                if encoding.shape[0] != 0:
                    attempts += 1
                    face_distances = fr.face_distance(cls.known_faces, encoding)

                    class_probs = scipy.special.softmax(face_distances ** -1)
                    log_probs = np.log(class_probs)
                    pred_idx = cls.known_classes[np.argmax(log_probs)]
                    # print(f"Person {person_idx} - {pred_idx} predicted")
                    if person_idx == pred_idx:
                        matches += 1

        print(f"Total accuracy: {(matches / attempts) * 100}%")

    @classmethod
    def label_data(cls):

        result_file = open("photo_norm-dist.txt", "w")

        for eval_file in tqdm(cls.EVAL_DIR.iterdir(), 'Label', len(list(cls.EVAL_DIR.iterdir())), unit='file'):
            if str(eval_file).__contains__('.DS_Store'):
                continue

            if not str(eval_file).endswith('.png'):
                continue

            encoding = cls.load_image(eval_file)

            if encoding.shape[0] != 0:
                face_distances = fr.face_distance(cls.known_faces, encoding)
                class_probs = scipy.special.softmax(face_distances ** -1)
                log_probs = np.log(class_probs)
                pred_class = cls.known_classes[np.argmax(log_probs)]

                sort_probs = {}

                for i in range(len(log_probs)):
                    sort_probs[int(cls.known_classes[i])] = log_probs[i]

                sort_probs = collections.OrderedDict(sorted(sort_probs.items()))

                res_line = '{0} {1} {2}\n'.format(os.path.basename(eval_file).replace('.png', ''), pred_class,
                                                  ' '.join(str(x) for x in sort_probs.values()))

                result_file.write(res_line)

        result_file.close()


recognizer = PhotoFaceRecognition()
recognizer.train_model()
recognizer.eval_model()
recognizer.label_data()
