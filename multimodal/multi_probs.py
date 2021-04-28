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
from sklearn.preprocessing import MinMaxScaler

sg = SpeechGaussian()
sg.train_gmm()

sg.train_path = Path('./tmp/train_big')

pr = PhotoFaceRecognition()
pr.train_model()

pr.TRAIN_DIR = Path('./dataset_clean/train')

DEV_DIR = Path('./dataset/dev2_full')
EVAL_DIR = Path('./dataset/eval')

identified_files = []

attempt = 0
accept = 0

for person_folder in tqdm(DEV_DIR.iterdir(), 'Eval', len(list(DEV_DIR.iterdir()))):
    if str(person_folder).__contains__('.DS_Store'):
        continue

    person_class = int(str(person_folder).split('/').pop())

    for person_file in person_folder.iterdir():
        attempt += 1
        strip_name = str(person_file).replace('.wav', '').replace('.png', '')

        face_file = f"{strip_name}.png"
        speech_file = f"{strip_name}.wav"

        sort_probs = {}

        encoding = pr.load_image(face_file)
        if encoding.shape[0] != 0:
            pred_class_face, log_probs_face = pr.eval_person(encoding)
            for i in range(len(log_probs_face)):
                sort_probs[int(pr.known_classes[i])] = log_probs_face[i]

        sort_probs = collections.OrderedDict(sorted(sort_probs.items()))

        pred_class_speech, log_probs_speech = sg.gmm_eval_speaker(speech_file)

        norm_probs_speech = scipy.special.softmax(log_probs_speech / sum(log_probs_speech))
        norm_probs_photo = scipy.special.softmax(np.array(list(sort_probs.values())))

        total_log_probs = norm_probs_photo * norm_probs_speech

        pred_class = np.argmax(total_log_probs) + 1
        # print(f"Person {person_class} - {pred_class} predicted")
        if pred_class == person_class:
            accept += 1

print(f"Total accuracy: {(accept / attempt) * 100}%")





