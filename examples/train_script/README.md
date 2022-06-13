## Before Running
```shell script
export PYTHONPATH=$PWD
```

## Get data label
```shell script
python labeler.py
```

## Run Training Script
```shell script
# --train_csv_path: specify the CSV path to training
# --cuda: using GPU
# --epochs: maximum number of epochs
# --lr: learning rate
# --eta_min: Min learning rate
# --weight_decay: weight decay rate
python main.py --train_csv_path data/some_image_list.csv --epochs 600 --cuda --save some_image_list
```
