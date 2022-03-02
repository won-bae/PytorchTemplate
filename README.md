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
To train a model, refer to the command below. Note that save_dir is the directory where all the checkpoints and logs are saved.
```bash
bash scripts/run.sh --config_path configs/default.yml --save_dir path/to/root
```

## Eval
To evaluate a model, refer to the command below. Note that save_dir should be syncronized with save_dir used for training.
```bash
bash scripts/run.sh --config_path configs/default.yml --save_dir path/to/dir --eval_only
```
## Citation
If you use this code, please cite:

    @Misc{bae2019pytorchtemplate,
      author = {Wonho Bae},
      title = {Pytorch Template},
      year = {2019},
      howpublished = "\url{https://github.com/won-bae/PytorchTemplate/}"
    }
