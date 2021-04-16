import os
import subprocess
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from pathlib import Path


class AudioTools:
    out_dir = Path('tmp')

    @classmethod
    def sox_speaker_remove_silence(cls, speaker_path):
        speaker_idx = str(speaker_path).split('/').pop()
        dst_dir = cls.out_dir.joinpath(speaker_idx)

        os.makedirs(dst_dir, exist_ok=True)

        for speaker_file in speaker_path.iterdir():
            if not str(speaker_file).endswith('.wav'):
                continue

            # tmp_file = Path(dst_dir).joinpath('tmp-{0}.wav'.format(speaker_idx))
            dest_file = dst_dir.joinpath(os.path.basename(speaker_file))

            sox_trim_ends = [
                "sox",
                str(speaker_file),
                str(dest_file),
                "silence",
                "1",
                "0.1",
                "1%",
                "reverse",
                "silence",
                "1",
                "0.1",
                "1%",
                "reverse"
            ]
            s = subprocess.call(sox_trim_ends)

            # sox_trim_inside = [
            #     "sox",
            #     str(tmp_file),
            #     str(dest_file),
            #     "silence",
            #     "1",
            #     "0.1",
            #     "1%",
            #     "-1",
            #     "0.1",
            #     "1%"
            # ]
            # s = subprocess.call(sox_trim_inside)
            #
            # os.remove(tmp_file)

    @classmethod
    def sox_prepare_dataset(cls, dataset_path, output_path=None, thread_count=4):
        if output_path is not None:
            cls.out_dir = output_path

        speakers = []

        for speaker_dir in dataset_path.iterdir():
            if str(speaker_dir).__contains__('.DS_Store'):
                continue
            speakers.append(speaker_dir)

        os.makedirs(cls.out_dir, exist_ok=True)

        with ThreadPool(thread_count) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.sox_speaker_remove_silence,
                        speakers
                    ),
                    'Sox',
                    len(speakers),
                    unit="speakers"
                )
            )

    @classmethod
    def sox_augument_speaker(cls, speaker_path):
        for speaker_file in speaker_path.iterdir():
            if not str(speaker_file).endswith('.wav'):
                continue

            dest_slow_file = Path(str(speaker_file).replace('.wav', '-s.wav'))

            sox_slow = [
                "sox",
                str(speaker_file),
                str(dest_slow_file),
                "tempo",
                "0.8"
            ]
            s = subprocess.call(sox_slow)

            dest_fast_file = Path(str(speaker_file).replace('.wav', '-f.wav'))

            sox_fast = [
                "sox",
                str(speaker_file),
                str(dest_fast_file),
                "tempo",
                "1.2"
            ]
            s = subprocess.call(sox_fast)

    @classmethod
    def sox_augument_dataset(cls, dataset_path, thread_count=4):
        speakers = []

        for speaker_dir in dataset_path.iterdir():
            if str(speaker_dir).__contains__('.DS_Store'):
                continue
            speakers.append(speaker_dir)

        with ThreadPool(thread_count) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.sox_augument_speaker,
                        speakers
                    ),
                    'Augument',
                    len(speakers),
                    unit="speakers"
                )
            )

    @classmethod
    def sox_remove_noise_speaker(cls, speaker_path):

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
    def sox_remove_noise_dataset(cls, dataset_path, thread_count=4):
        speakers = []

        for speaker_dir in dataset_path.iterdir():
            if str(speaker_dir).__contains__('.DS_Store'):
                continue
            speakers.append(speaker_dir)

        with ThreadPool(thread_count) as pool:
            list(
                tqdm(
                    pool.imap(
                        cls.sox_remove_noise_speaker,
                        speakers
                    ),
                    'Remove noise',
                    len(speakers),
                    unit="speakers"
                )
            )
