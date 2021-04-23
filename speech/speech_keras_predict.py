from pathlib import Path
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
import speech_keras_config as Config
import speech_keras_data_man as dm
import speech_keras_model as km

print(f"Restoring model {Config.model_save_filename}")
model = keras.models.load_model(Config.model_save_filename)

print(f"Restoring class names from {Config.class_names_savefile}")
class_names = np.load(Config.class_names_savefile)
# EVAL_DIR = Path('./dataset/dev2')
# AudioTools.sox_prepare_dataset(EVAL_DIR, Path('./dataset/dev2_proc'))

EVAL_DIR = Path('../dataset/dev2')

attempts = 0
true_accept = 0

for eval_speaker in tqdm(EVAL_DIR.iterdir(), 'Eval', len(list(EVAL_DIR.iterdir())), unit='speakers'):

    speaker_idx = int(str(eval_speaker).split('/').pop())

    for speaker_file in eval_speaker.iterdir():
        if not str(speaker_file).endswith('.wav'):
            continue

        samples, sampling_rate = tf.audio.decode_wav(
            tf.io.read_file(str(speaker_file)), desired_channels=1
        )
        if sampling_rate == Config.sampling_rate:
            # Number of slices of 16000 each that can be generated from the noise sample
            slices = int(samples.shape[0] / Config.sampling_rate)
            try:
                samples = tf.split(samples[: slices * Config.sampling_rate], slices)
                segment_ffts = dm.audio_to_fft(samples)
                y_pred = model.predict(segment_ffts)
                y_classes = np.argmax(y_pred, axis=-1)
                pred_class = int(class_names[np.bincount(y_classes).argmax()])
                # print(f"Speaker: {speaker_idx} - Predicted: {pred_class}")

                if pred_class == speaker_idx:
                    true_accept += 1
            except:
                print(str(speaker_file))
        else:
            print("Sampling rate for {} is incorrect. Ignoring it".format(str(speaker_file)))
            continue

        attempts += 1

acc = true_accept / attempts
print(f"Total accuracy: {acc * 100}%")