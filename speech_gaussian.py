from ikrlib import wav16khz2mfcc
from sklearn.mixture import GaussianMixture
from pathlib import Path

class_cnt = 0
gaussian_cnt = 10

gmm_arr = []

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

for dir_name in train_path.iterdir():
    if dir_name == '.DS_Store':
        continue

    class_cnt += 1
    cls_td = wav16khz2mfcc(str(dir_name))
    print(cls_td)
    # gmm = GaussianMixture(gaussian_cnt).fit(cls_td)
    # gmm_arr.append(gmm)


print(gmm_arr)