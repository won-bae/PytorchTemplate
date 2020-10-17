# PytorchTemplate

This is a template for a deep learning project based on pytorch.


## Requirements
To fulfill the basic requirements, run the command below.
```bash
pip install -r requirements.txt
```

## Configs
Configs are written in the form of yaml. Please refer to the configs/default.yml for the details about how to structure configs.

## Datasets
It assumes datasets are located in data root which can be modified in the data section of a config file.

## Train
To train a model, refer to the command below. Note that train_root/tag is defined as train_dir where all the checkpoints and logs are saved.
```bash
bash scripts/run_train.sh --config_path configs/default.yml --train_root path/to/root --tag vgg16
```

## Eval
To evaluate a model, refer to the command below. Note that train_dir should be syncronized with train_root/tag used for training.
```bash
bash scripts/run_eval.sh --config_path configs/default.yml --train_dir path/to/dir --tag vgg16_test
```
