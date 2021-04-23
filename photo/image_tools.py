from multiprocessing.pool import ThreadPool
import face_frontalization.frontalize as frontalize
import face_frontalization.facial_feature_detector as feature_detection
import face_frontalization.camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class ImageTools:
    model_path = os.path.dirname(os.path.abspath(__file__))
    img_width = 160

    front_nonsym_suffix = '-f.png'

    @classmethod
    def crop_frontalized(cls, im):
        width, height = im.size
        new_width = cls.img_width
        left = (width - new_width) / 2
        top = (height - new_width) / 2
        right = (width + new_width) / 2
        bottom = (height + new_width) / 2

        return im.crop((left, top, right, bottom))

    @classmethod
    def frontalize_face(cls, img_data):
        img_path = img_data['img_path']
        result_path = img_data['result_path']

        os.makedirs(result_path, exist_ok=True)

        model3D = frontalize.ThreeD_Model(cls.model_path + "/face_frontalization/frontalization_models/model3Ddlib.mat",
                                          'model_dlib')

        img = cv2.imread(str(img_path), 1)
        lmarks = feature_detection.get_landmarks(img)
        try:
            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
        except IndexError:
            return
        # load mask to exclude eyes from symmetry
        eyemask = np.asarray(io.loadmat('../face_frontalization/frontalization_models/eyemask.mat')['eyemask'])
        # perform frontalization
        frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)

        img_basename = os.path.basename(img_path)
        raw_name = result_path.joinpath(img_basename.replace('.png', '-f.png'))

        im_raw = Image.fromarray(frontal_raw[:, :, ::-1])
        im_raw = cls.crop_frontalized(im_raw)
        im_raw.save(raw_name)

        sym_name = result_path.joinpath(img_basename.replace('.png', '-fs.png'))

        im_sym = Image.fromarray(frontal_sym[:, :, ::-1])
        im_sym = cls.crop_frontalized(im_sym)
        im_sym.save(sym_name)


    @classmethod
    def frontalize_dataset(cls, dataset_src, output_dir, thread_count=4):
        img_datas = []

        for data_dir in dataset_src.iterdir():
            if str(data_dir).__contains__('.DS_Store'):
                continue

            speaker_idx = str(data_dir).split('/').pop()
            data_output = output_dir.joinpath(speaker_idx)

            for data_file in data_dir.iterdir():
                if not (str(data_file).endswith('.png')):
                    continue

                img_data = {
                    'img_path': data_file,
                    'result_path': data_output
                }
                img_datas.append(img_data)

        os.makedirs(output_dir, exist_ok=True)

        with ThreadPool(thread_count) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.frontalize_face,
                        img_datas
                    ),
                    'Frontalize',
                    len(img_datas),
                    unit="face"
                )
            )

    @classmethod
    def flip_face(cls, img_path):
        new_name = Path(str(img_path).replace(cls.front_nonsym_suffix, '-fs_flip.png'))

        im = Image.open(img_path)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im.save(new_name)

    @classmethod
    def augument_dataset(cls, dataset_src, thread_count=4):
        faces = []

        for image_dir in dataset_src.iterdir():
            if str(image_dir).__contains__('.DS_Store'):
                continue

            for image in image_dir.iterdir():
                if str(image).endswith(cls.front_nonsym_suffix):
                    faces.append(image)

        with ThreadPool(thread_count) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.flip_face,
                        faces
                    ),
                    'Augument',
                    len(faces),
                    unit="face"
                )
            )