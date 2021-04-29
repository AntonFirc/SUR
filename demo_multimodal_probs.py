import sys
import os
sys.path.append(os.path.abspath('./multimodal'))
from multi_probs import MultimodalProbability
sys.path.append(os.path.abspath('./speech'))
from audio_tools import AudioTools
import subprocess

rm_tmp = [
    'rm',
    '-rf',
    './temp',
]
subprocess.call(rm_tmp)

multi = MultimodalProbability()
at = AudioTools()

at.process_dataset(at.sox_remove_silence, multi.sg.train_orig_path, output_path=multi.sg.train_path)
at.process_dataset(at.sox_augument_data, multi.sg.train_path, aug_options=[0.9, 0.95, 1.05, 1.1])
at.process_dataset(at.sox_remove_silence, multi.sg.eval_orig_path, output_path=multi.sg.eval_path, eval_dataset=True)

multi.train_model()

multi.eval_model()

multi.label_data()