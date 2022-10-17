import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log, mkdir_p
from src.utils.config import Config

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', required=True,
                        help="Path to a config")
    parser.add_argument('-r', '--save_root', default='save_dir',
                        help='Path to dir to save train dirs')
    parser.add_argument('-t', '--tag', required=True,
                        help='Tag for a run')
    parser.add_argument('-e', '--eval_only',  action='store_true', default=False,
                        help="evaluate only if it is true")
    args = parser.parse_args()

    # Generate a save dir
    save_dir = os.path.join(args.save_root, args.tag)
    mkdir_p(save_dir)

    # Load a logger
    logger = load_log(save_dir)

    # Generate a config
    config = Config.fromfile(args.config_path)
    config.dump(os.path.join(save_dir, "config.py"))

    engine = Engine(config=config, logger=logger, save_dir=save_dir)

    if args.eval_only:
        engine.evaluate()
    else:
        engine.run()
