import face_recognition as fr
import numpy as np
from pathlib import Path
from tqdm import tqdm

TRAIN_DIR = Path('./dataset_clean/train')
# DEV_DIR = Path('./dataset_clean/train')
DEV_DIR = Path('./dataset/dev')

known_faces = []

known_classes = []

for person_folder in tqdm(TRAIN_DIR.iterdir(), 'Train', len(list(TRAIN_DIR.iterdir())), unit='person'):

    if str(person_folder).__contains__('.DS_Store'):
        continue

    person_idx = int(str(person_folder).split('/').pop())
    person_faces = []

    for person_file in person_folder.iterdir():
        if not str(person_file).endswith('.png'):
            continue

        img = fr.load_image_file(person_file)
        locations = fr.face_locations(img, number_of_times_to_upsample=3, model="cnn")
        encoding = np.array(fr.face_encodings(img, known_face_locations=locations, num_jitters=10, model="large"))

        if encoding.shape[0] != 0:
            person_faces.append(encoding[0])

    known_classes.append(person_idx)
    known_faces.append(np.mean(person_faces, axis=0))

attempts = 0
matches = 0

for dev_folder in tqdm(DEV_DIR.iterdir(), 'Eval', len(list(DEV_DIR.iterdir())), unit='person'):
    if str(dev_folder).__contains__('.DS_Store'):
        continue

    person_idx = int(str(dev_folder).split('/').pop())

    for dev_file in dev_folder.iterdir():
        if not str(dev_file).endswith('.png'):
            continue

        img = fr.load_image_file(dev_file)
        locations = fr.face_locations(img, number_of_times_to_upsample=3, model="cnn")
        encoding = np.array(fr.face_encodings(img, known_face_locations=locations, num_jitters=10, model="large"))

        if encoding.shape[0] != 0:
            attempts += 1
            face_distances = fr.face_distance(known_faces, encoding)

            pred_idx = known_classes[np.argmin(face_distances)]
            # print(f"Person {person_idx} - {pred_idx} predicted")
            if person_idx == pred_idx:
                matches += 1

print(f"Total accuracy: {(matches / attempts) * 100}%")
