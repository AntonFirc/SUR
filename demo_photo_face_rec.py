import sys
import os
sys.path.append(os.path.abspath('./photo'))
from photo_face_recognition import PhotoFaceRecognition

pr = PhotoFaceRecognition()

pr.train_model()

pr.eval_model()

pr.label_data()
