import distutils
from distutils import dir_util

distutils.dir_util.copy_tree('./dataset/train', './train_big/')
distutils.dir_util.copy_tree('./dataset/dev', './train_big/')
