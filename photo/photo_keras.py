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

from PIL import Image


dev_path = Path('../dataset/dev')
train_path = Path('../dataset/train')
dev_path2 = Path('./2dataset/dev')
train_path2 = Path('./2dataset/train')

class_cnt = 31


def load_faces(dir_path):
    participants = []

    for i in range(class_cnt):
        participant_dir = dir_path.joinpath(str(i + 1))
        participants.append(participant_dir)

    feature_list = []
    for participant_dir in participants:
        participant_idx = int(str(participant_dir).split('/').pop())
        photo_features = []
        for f in glob(str(participant_dir) + '/*.png'):
            image = imread(f, True).astype(np.float64)
            photo_features.append(image)
        feature_list.insert(participant_idx, photo_features)

    img_cnt = len(feature_list[0])
    faces = np.array(feature_list).reshape((img_cnt * class_cnt, 224, 224, 3))
    return faces


def store_image(url, local_file_name):
  with urllib.request.urlopen(url) as resource:
    with open(local_file_name, 'wb') as f:
      f.write(resource.read())


def get_model_scores(faces):
    samples = np.asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    # model = VGGFace(model='resnet50',
    #   include_top=False,
    #   input_shape=(80, 80, 3),
    #   pooling='avg')
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

    # perform prediction
    return model.predict(samples)

# facesX = load_faces(dev_path2)
facesX = load_faces(train_path2)
facesY = load_faces(train_path2)

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
    print(idx, np_s.argmin() // 6, scores[np_s.argmin()])
    all += 1
    # indexing
    if idx // 6 == np_s.argmin() // 6:
        good += 1
    # if score <= 0.4:
      # Printing the IDs of faces and score
      # print(idx, idy, score)
      # Displaying each matched pair of faces
      # plt.imshow(facesX[idx])
      # plt.show()
      # plt.imshow(facesY[idy])
      # plt.show()

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
