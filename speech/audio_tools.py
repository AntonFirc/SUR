import os
import subprocess
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from pathlib import Path


class AudioTools:
    out_dir = Path('../tmp')
    aug_options = [0.9, 1.1]

    @classmethod
    def sox_remove_silence(cls, speaker_path):
        speaker_idx = str(speaker_path).split('/').pop()

        if speaker_idx == 'eval':
            dst_dir = cls.out_dir
        else:
            dst_dir = cls.out_dir.joinpath(speaker_idx)

        os.makedirs(dst_dir, exist_ok=True)

        for speaker_file in speaker_path.iterdir():
            if not str(speaker_file).endswith('.wav'):
                continue

            dest_file = dst_dir.joinpath(os.path.basename(speaker_file))

            # translates to "trim all silence until 0.1s of sound is detected,
            # then trim all silence after 0.7s of silence is detected and repeat
            # TLDR: removes silence on the beginning and end of recording, and shortens long pauses in the middle
            sox_trim_inside = [
                "sox",
                str(speaker_file),
                str(dest_file),
                "silence",
                "1",
                "0.1",
                "1%",
                "-1",
                "0.7",
                "1%"
            ]
            s = subprocess.call(sox_trim_inside)

    @classmethod
    def sox_augument_data(cls, speaker_path):
        for speaker_file in speaker_path.iterdir():
            if not str(speaker_file).endswith('.wav'):
                continue

            for option in cls.aug_options:
                dest_file = Path(str(speaker_file).replace('.wav', '-{0}.wav'.format(str(option))))

                sox_slow = [
                    "sox",
                    str(speaker_file),
                    str(dest_file),
                    "tempo",
                    str(option)
                ]
                s = subprocess.call(sox_slow)


    @classmethod
    def sox_remove_noise(cls, speaker_path):

        for speaker_file in speaker_path.iterdir():
            if not str(speaker_file).endswith('.wav'):
                continue

            tmp_file = Path(str(speaker_file).replace('.wav', '-noise.prof'))

            sox_noise_prof = [
                "sox",
                str(speaker_file),
                "-n",
                "noiseprof",
                str(tmp_file)
            ]
            s = subprocess.call(sox_noise_prof)

            dest_file = Path(str(speaker_file).replace('.wav', '-d.wav'))

            sox_noise_remove = [
                "sox",
                str(speaker_file),
                str(dest_file),
                "noisered",
                str(tmp_file),
                "0.2"
            ]
            s = subprocess.call(sox_noise_remove)

            os.remove(tmp_file)
            os.remove(speaker_file)


    @classmethod
    def sox_split_audio(cls, speaker_path):

        for speaker_file in speaker_path.iterdir():
            if not str(speaker_file).endswith('.wav'):
                continue

            tmp_file = Path(str(speaker_file).replace('.wav', '1s.wav'))

            sox_split = [
                "sox",
                str(speaker_file),
                str(tmp_file),
                "trim",
                "0",
                "1",
                ":",
                "newfile",
                ":",
                "restart",
            ]
            s = subprocess.call(sox_split)

            os.remove(speaker_file)

    @classmethod
    def process_dataset(cls, task, dataset_path, task_name="Process", output_path=None, thread_count=4, unit='speaker', aug_options=None, eval_dataset=False):
        assert task is not None

        if output_path is not None:
            cls.out_dir = output_path

        if aug_options is not None:
            cls.aug_options = aug_options

        speakers = []

        if eval_dataset:
            speakers.append(dataset_path)
        else:
            for speaker_dir in dataset_path.iterdir():
                if str(speaker_dir).__contains__('.DS_Store'):
                    continue
                speakers.append(speaker_dir)

        with ThreadPool(thread_count) as pool:
            list(
                tqdm(
                    pool.imap(
                        task,
                        speakers
                    ),
                    task_name,
                    len(speakers),
                    unit=unit
                )
            )
