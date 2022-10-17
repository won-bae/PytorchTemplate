import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log, mkdir_p, split_dict, dict_to_str
from src.utils.config import Config

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_path', required=True,
                        help="Path to a config")
    parser.add_argument('-s', '--sweep_path', required=True,
                        help="Path to a sweep config")
    parser.add_argument('-r', '--save_root', default='save_dir',
                        help='Path to dir to save train dirs')
    parser.add_argument('-t', '--tag', required=True,
                        help='Tag for a run')
    parser.add_argument('-e', '--eval_only',  action='store_true', default=False,
                        help="Evaluate only if it is true")
    args = parser.parse_args()

    # Generate a save dir
    save_dir = os.path.join(args.save_root, args.tag)
    mkdir_p(save_dir)

    # Load a logger
    logger = load_log(save_dir)

    # Generate a config
    config = Config.fromfile(args.config_path)
    sweep_config_all = Config.fromfile(args.sweep_path)
    sweep_config_all.dump(os.path.join(save_dir, "sweep.py"))

    sweep_dict_list = split_dict(dict(sweep_config_all))

    for sweep_dict in sweep_dict_list:
        sweep_config = Config(sweep_dict)

        subdir_name = dict_to_str(sweep_dict)
        sub_dir = os.path.join(save_dir, subdir_name)

        config.merge_from_dict(sweep_config)

        mkdir_p(sub_dir)
        config.dump(os.path.join(sub_dir, "config.py"))

        engine = Engine(config=config, logger=logger, save_dir=sub_dir)

        if args.eval_only:
            engine.evaluate()
        else:
            engine.run()
