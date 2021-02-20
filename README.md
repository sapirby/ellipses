# Ellipses

## Preparation

Install PyTorch (stable version 1.7.1), tensorboard, pandas.

## Train

Train from scratch:
```
python train.py --data_dir [your data path]
```

Model files, a log file and tensorboard events file are saved locally.

Alternately, download a [pretrained model](https://drive.google.com/file/d/1wQjoAF-XruySRxAE2DsmXX5mwDAvBc6I/view?usp=sharing).

## Test

```
python test.py --data_dir [your data path] --model_load_path [your *.pt model file path]
```

You should get: accuracy: 98.60, mean absolute errors: center_x 0.86, center_y 0.85, axis_1 1.26, axis_2 0.98, angle 14.67, angle error 8.15%.


## Report
See report [here](https://docs.google.com/document/d/1I2Kem6zZDfrQes4q6sQ1vXKhbp0DYUlCr3CXT6wFhdI/edit?usp=sharing).
