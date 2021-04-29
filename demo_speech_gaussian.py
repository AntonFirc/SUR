import subprocess
import sys
import os

sys.path.append(os.path.abspath('./speech'))
from speech_gaussian import SpeechGaussian
from audio_tools import AudioTools

rm_tmp = [
    'rm',
    '-rf',
    './temp',
]
subprocess.call(rm_tmp)

sg = SpeechGaussian()
at = AudioTools()

# remove silence from train and dev data
at.process_dataset(at.sox_remove_silence, sg.train_orig_path, output_path=sg.train_path)
at.process_dataset(at.sox_remove_silence, sg.dev_orig_path, output_path=sg.dev_path)

# augument train data
at.process_dataset(at.sox_augument_data, sg.train_path, aug_options=[0.9, 0.95, 1.05, 1.1])

# remove silence from eval data
at.process_dataset(at.sox_remove_silence, sg.eval_orig_path, output_path=sg.eval_path, eval_dataset=True)

# train GMM
sg.train_gmm()
# evaluate on dev data
sg.gmm_evaluate_model()
# label eval data and output result file
sg.gmm_label_data(sg.eval_path)
