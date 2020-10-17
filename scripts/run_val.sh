#!/bin/sh

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -c|--config_path)
      CONFIG_PATH="$2"
      shift # past argument
      ;;
      -t|--_tag)
      TAG="$2"
      shift # past argument
      ;;
      -d|--train_dir)
      DIR="$2"
      shift
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

python val.py \
    --config_path=${CONFIG_PATH} \
    --tag=${TAG} \
    --train_dir=${DIR}
