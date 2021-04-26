from glob import glob
import tensorflow as tf
from keras import applications
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from scipy.spatial.distance import cosine
import urllib.request
from pathlib import Path
from matplotlib import pyplot as plt
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image

from PIL import Image


dev_path = Path('../dataset/dev')
train_path = Path('../dataset/train')
dev_path2 = Path('../2dataset/dev')
train_path2 = Path('../2dataset/train')

class_cnt = 31


def extract_face_from_image(image_path, detector, required_size=(224, 224)):
  # load image and detect faces
    image = plt.imread(image_path)
    faces = detector.detect_faces(image)

    face_image = None

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_image = face_array

    return face_image


def load_faces(dir_path):
    participants = []

    for i in range(class_cnt):
        participant_dir = dir_path.joinpath(str(i + 1))
        participants.append(participant_dir)

    feature_list = []
    # detector = MTCNN()
    for participant_dir in participants:
        participant_idx = int(str(participant_dir).split('/').pop())
        photo_features = []
        for f in glob(str(participant_dir) + '/*.png'):
            # image = Image.open(f)
            # image = extract_face_from_image(f, detector=detector)
            image = imread(f)
            photo_features.append(image)
        feature_list.insert(participant_idx, photo_features)

    img_cnt = len(feature_list[0])
    images = np.array(feature_list).reshape((img_cnt * class_cnt, 80, 80, 3))

    return images


def get_model_scores(faces):
    samples = np.asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    # model = VGGFace(model='resnet50',
    #   include_top=False,
    #   input_shape=(80, 80, 3),
    #   pooling='avg')
    model = VGGFace(model='vgg16',
      include_top=False,
      weights='vggface',
      input_shape=(80, 80, 3),
      pooling='avg'
        )

    # perform prediction
    return model.predict(samples)


facesX = load_faces(dev_path)
# facesX = load_faces(train_path2)
facesY = load_faces(train_path)

model_scores_xi = get_model_scores(facesX)
model_scores_yi = get_model_scores(facesY)

good = 0
all = 0
for idx, face_score_1 in enumerate(model_scores_xi):
    scores = []
    for idy, face_score_2 in enumerate(model_scores_yi):
        score = cosine(face_score_1, face_score_2)
        scores.append(score)

    np_s = np.array(scores)
    # np_s_min = np_s.argsort()[:class_cnt]
    # np_s_min = np_s_min // 6
    # counts = np.bincount(np_s_min)
    # values, counts = np.unique(np_s_min, return_counts=True)
    # ind = np.argmax(counts)
    # np_s_sum = np.add.reduceat(np_s, np.arange(0, len(np_s), 6))

    ind = np.argmin(np_s)

    print(idx // 2 + 1, ind // 6 + 1)
    all += 1
    # indexing
    if idx // 2 == ind // 6:
        good += 1

print((good / all) * 100, "% accuracy")
# resize images
# participants = []
#
# for i in range(class_cnt):
#     participant_dir = train_path.joinpath(str(i + 1))
#     participants.append(participant_dir)
#
# feature_list = []
# for participant_dir in participants:
#     participant_idx = int(str(participant_dir).split('/').pop())
#     for f in glob(str(participant_dir) + '/*.png'):
#         image = Image.open(f)
#         image = image.resize((224, 224))
#         image.save("2" + f)
#
# participants = []
#
# for i in range(class_cnt):
#     participant_dir = dev_path.joinpath(str(i + 1))
#     participants.append(participant_dir)
#
# feature_list = []
# for participant_dir in participants:
#     participant_idx = int(str(participant_dir).split('/').pop())
#     for f in glob(str(participant_dir) + '/*.png'):
#         image = Image.open(f)
#         image = image.resize((224, 224))
#         image.save("2" + f)
