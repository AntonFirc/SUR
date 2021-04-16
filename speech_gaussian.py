from ikrlib import wav16khz2mfcc
from sklearn.mixture import GaussianMixture
from pathlib import Path
import matplotlib

class_cnt = 31
gaussian_cnt = 10

gmm_arr = []

dev_path = Path('./dataset/dev')
train_path = Path('./dataset/train')

for i in range(class_cnt):
    print(train_path.iterdir())
    #cls_td = wav16khz2mfcc()
    #gmm = GaussianMixture(gaussian_cnt).fit(cls_td)