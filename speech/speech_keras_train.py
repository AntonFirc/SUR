import os
import math
import numpy as np
# use tensorflow==2.4 or tensorflow-gpu==2.4
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tensorflow.keras.optimizers import SGD
import speech_keras_config as Config
import speech_keras_data_man as dm
import speech_keras_model as km

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

DATASET_ROOT = Path('./dataset_clean')

AUDIO_DIR = DATASET_ROOT.joinpath('train_biger_aug')
NOISE_DIR = DATASET_ROOT.joinpath('noise')

DATA_SPLIT = Config.data_spit
SHUFFLE_SEED = Config.shuffle_seed
SAMPLING_RATE = Config.sampling_rate
SCALE = Config.scale
BATCH_SIZE = Config.batch_size
EPOCHS = Config.epochs

# Get the list of all noise files
noise_paths = []
for subdir in os.listdir(NOISE_DIR):
    subdir_path = Path(NOISE_DIR) / subdir
    if os.path.isdir(subdir_path):
        noise_paths += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]

print(
    "Found {} files belonging to {} directories".format(
        len(noise_paths), len(os.listdir(NOISE_DIR))
    )
)

command = (
        "for dir in `ls -1 " + str(NOISE_DIR) + "`; do "
                                                "for file in `ls -1 " + str(NOISE_DIR) + "/$dir/*.wav`; do "
                                                                                         "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
                                                                                         "$file | grep sample_rate | cut -f2 -d=`; "
                                                                                         "if [ $sample_rate -ne 16000 ]; then "
                                                                                         "ffmpeg -hide_banner -loglevel panic -y "
                                                                                         "-i $file -ar 16000 temp.wav; "
                                                                                         "mv temp.wav $file; "
                                                                                         "fi; done; done"
)

os.system(command)


noises = []
for path in noise_paths:
    sample = dm.load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
    )
)

# Get the list of audio file paths along with their corresponding labels
class_names = os.listdir(AUDIO_DIR)
print("Our class names: {}".format(class_names, ))
print(f"Saving class names to {Config.class_names_savefile}")
np.save(Config.class_names_savefile, class_names)

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name, ))
    dir_path = Path(AUDIO_DIR) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

# Shuffle
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(DATA_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 2 datasets, one for training and the other for validation
train_ds = dm.paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = dm.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

if Config.add_noise:
    print("Adding noise to samples")
    # Add noise to the training set
    train_ds = train_ds.map(
        lambda x, y: (dm.add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

# Transform audio wave to the frequency domain using `audio_to_fft`
train_ds = train_ds.map(
    lambda x, y: (dm.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (dm.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

model_save_filename = Config.model_save_filename
initial_lr = Config.initial_lr

if os.path.exists(model_save_filename):
    print(f"Restoring model {model_save_filename}")
    model = keras.models.load_model(model_save_filename)
else:
    print(f"Building new model in {model_save_filename}")
    model = km.build_model((SAMPLING_RATE // 2, 1), len(class_names))

model.summary()

optimizer = SGD(learning_rate=initial_lr, momentum=0.4)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


def scheduler(epoch, lr):
    drop = 0.5
    epochs_drop = 10.0
    new_lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return new_lr


earlystopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)
scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    batch_size=BATCH_SIZE,
    callbacks=[mdlcheckpoint_cb, earlystopping_cb, scheduler_cb],
)


