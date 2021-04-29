# SUR project - indentification system

Project contains implemented identification systems for person identification using speech or face.

Run demo scripts to classify provided data:
- `python3 demo_speech_gaussian.py`
- `python3 demo_speech_keras.py`
- `python3 demo_photo_face_rec.py`
- `python3 demo_multimodal_probs.py`

Naming of the components follows this scheme: modality_technology
i.e. `speech_gaussian` -> speech identification using gaussian mixture models

## Requirements
- `pip install -r requirements.txt`
- `conda install -c conda-forge imread`
- `conda install opencv`
- `conda install nomkl`

## Metacentrum cheatsheet
- `module add anaconda3-2019.10`
- `module add sox-14.4.2`
- `module add cuda-10.1`
- `module add cudnn/cudnn-8.0.4.30-11.0-linux-x64-intel-19.0.4-fvpdtul`