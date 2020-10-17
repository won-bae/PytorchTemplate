import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log, mkdir_p

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--tag', default='',
                        help="tag to discern evaluation instances")
    parser.add_argument('--train_dir', default='',
                        help="Path to train dir")
    args = parser.parse_args()

    eval_dir = mkdir_p(
        os.path.join(args.train_dir, 'val_' + args.tag))

    eval_dir_name = '/'.join(eval_dir.split('/')[-2:])
    log = load_log(eval_dir_name)

    if not args.config_path:
        args.config_path = os.path.join(args.train_dir, "config.yml")
        log.warning("config_path is not given: use [%s] as default.", args.config_path)
    shutil.copyfile(args.config_path, os.path.join(eval_dir, "config.yml"))

    engine = Engine(mode='val', config_path=args.config_path, log=log,
                    train_dir=args.checkpoint_dir, eval_dir=eval_dir)
    engine.evaluate()
