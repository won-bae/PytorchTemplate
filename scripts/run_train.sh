#!/bin/sh

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -c|--config_path)
      CONFIG_PATH="$2"
      shift # past argument
      ;;
      -r|--train_root)
      TRAIN_ROOT="$2"
      shift # past argument
      ;;
      -t|--tag)
      TAG="$2"
      shift # past argument
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

python train.py \
    --config_path=${CONFIG_PATH} \
    --train_root=${TRAIN_ROOT} \
    --tag=${TAG}
