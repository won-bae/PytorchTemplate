import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--train_root', default='',
                        help='Path to dir to save train dirs')
    parser.add_argument('--tag', default='',
                        help="tag to discern training instances")
    args = parser.parse_args()

    train_dir = os.path.join(args.train_root, args.tag)
    log = load_log(args.tag)

    shutil.copyfile(args.config_path, os.path.join(train_dir, "config.yml"))

    engine = Engine(
        mode='train', config_path=args.config_path, log=log, train_dir=train_dir)
    engine.train()
