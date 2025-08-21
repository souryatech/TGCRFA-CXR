import numpy as np

import os

train_imgs = []
for i in os.listdir('datasets/vindr/val'):
    print(i)
    train_imgs.append(i)
print(len(train_imgs))