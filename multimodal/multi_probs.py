import sys
import os

sys.path.append(os.path.abspath('./speech'))
sys.path.append(os.path.abspath('./photo'))
from speech_gaussian import SpeechGaussian
from photo_face_recognition import PhotoFaceRecognition
from pathlib import Path
import collections
import numpy as np
from tqdm import tqdm
import scipy.special


class MultimodalProbability:
    sg = SpeechGaussian()
    pr = PhotoFaceRecognition()

    DEV_DIR = Path('./dataset/dev')
    EVAL_DIR = Path('./dataset/eval')

    identified_files = []

    @classmethod
    def train_model(cls):
        cls.sg.train_gmm()
        cls.pr.train_model()

    @classmethod
    def eval_model(cls):
        attempt = 0
        accept = 0

        for person_folder in tqdm(cls.DEV_DIR.iterdir(), 'Eval', len(list(cls.DEV_DIR.iterdir()))):
            if str(person_folder).__contains__('.DS_Store'):
                continue

            person_class = int(str(person_folder).split('/').pop())

            for person_file in person_folder.iterdir():
                attempt += 1
                strip_name = str(person_file).replace('.wav', '').replace('.png', '')

                face_file = f"{strip_name}.png"
                speech_file = f"{strip_name}.wav"

                sort_probs = {}

                encoding = cls.pr.load_image(face_file)
                if encoding.shape[0] != 0:
                    pred_class_face, log_probs_face = cls.pr.eval_person(encoding)
                    for i in range(len(log_probs_face)):
                        sort_probs[int(cls.pr.known_classes[i])] = log_probs_face[i]

                sort_probs = collections.OrderedDict(sorted(sort_probs.items()))

                pred_class_speech, log_probs_speech = cls.sg.gmm_eval_speaker(speech_file)

                norm_probs_speech = scipy.special.softmax(log_probs_speech / sum(log_probs_speech))
                norm_probs_photo = scipy.special.softmax(np.array(list(sort_probs.values())))

                total_log_probs = norm_probs_photo * norm_probs_speech

                pred_class = np.argmax(total_log_probs) + 1
                # print(f"Person {person_class} - {pred_class} predicted")
                if pred_class == person_class:
                    accept += 1

        print(f"Total accuracy: {(accept / attempt) * 100}%")

    @classmethod
    def label_data(cls):
        result_file = open("multimodal_probabilities.txt", "w")

        for eval_file in tqdm(cls.EVAL_DIR.iterdir(), 'Label', len(list(cls.EVAL_DIR.iterdir())), 'file'):
            strip_name = str(eval_file).replace('.wav', '').replace('.png', '')

            face_file = f"{strip_name}.png"
            speech_file = f"{strip_name}.wav"

            sort_probs = {}

            encoding = cls.pr.load_image(face_file)
            if encoding.shape[0] != 0:
                pred_class_face, log_probs_face = cls.pr.eval_person(encoding)
                for i in range(len(log_probs_face)):
                    sort_probs[int(cls.pr.known_classes[i])] = log_probs_face[i]

            sort_probs = collections.OrderedDict(sorted(sort_probs.items()))

            pred_class_speech, log_probs_speech = cls.sg.gmm_eval_speaker(speech_file)

            norm_probs_speech = scipy.special.softmax(log_probs_speech / sum(log_probs_speech))
            norm_probs_photo = scipy.special.softmax(np.array(list(sort_probs.values())))

            total_log_probs = np.log(norm_probs_photo * norm_probs_speech)

            pred_class = np.argmax(total_log_probs) + 1

            res_line = '{0} {1} {2}\n'.format(os.path.basename(eval_file).replace('.wav', ''), pred_class,
                                              ' '.join(str(x) for x in total_log_probs))
            result_file.write(res_line)

        result_file.close()

