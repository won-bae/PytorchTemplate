import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log, mkdir_p

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--save_dir', default='',
                        help='Path to dir to save train dirs')
    parser.add_argument('--eval_only',  action='store_true', default=False,
                        help="evaluate only if it is true")
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir)
    mkdir_p(save_dir)
    logger = load_log(save_dir)

    shutil.copyfile(args.config_path, os.path.join(save_dir, "config.yml"))

    engine = Engine(config_path=args.config_path, logger=logger, save_dir=save_dir)

    if args.eval_only:
        engine.evaluate()
    else:
        engine.run()
