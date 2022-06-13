import glob
from pathlib import Path

import pandas

data_root = Path.home() / '.alaas/data/CIFAR10'
train_root = data_root / 'train'
test_root = data_root / 'test'

### Train data
image_paths = list(glob.glob(str(train_root / '*/*.jpg')))
labels = list()
for image_path in image_paths:
    labels.append(image_path.split('/')[-2])


df = pandas.DataFrame(zip(image_paths, labels), columns=('image_paths', 'labels'))
df.to_csv('train_data.csv', index=False)

### Test data
image_paths = list(glob.glob(str(test_root / '*/*.jpg')))
labels = list()
for image_path in image_paths:
    labels.append(image_path.split('/')[-2])


df = pandas.DataFrame(zip(image_paths, labels), columns=('image_paths', 'labels'))
df.to_csv('test_data.csv', index=False)
