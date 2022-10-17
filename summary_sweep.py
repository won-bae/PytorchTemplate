import argparse
import os
import json
import shutil
import pandas as pd
from src.engine import Engine
from src.utils.util import split_dict, dict_to_str, flatten_dict, get_project_root
from src.utils.config import Config

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--save_root', default='save_dir',
                        help='Path to dir to save train dirs')
    parser.add_argument('-t', '--tag', required=True,
                        help='Tag for a run')
    args = parser.parse_args()

    root_dir = get_project_root()
    save_dir = os.path.join(root_dir, args.save_root, args.tag)
    sweep_path = os.path.join(save_dir, "sweep.py")

    sweep_config_all = Config.fromfile(sweep_path)
    sweep_dict_list = split_dict(dict(sweep_config_all))

    metric = Config.fromfile(
        os.path.join(save_dir, dict_to_str(sweep_dict_list[0]), 'config.py'))['eval']['standard']

    columns = [key.split('.')[-1] for key in flatten_dict(sweep_dict_list[0]).keys()]
    map_column_to_full = {
        key.split('.')[-1]: key for key in flatten_dict(sweep_dict_list[0]).keys()}
    df = pd.DataFrame(columns=columns + [metric])

    for sweep_dict in sweep_dict_list:
        subdir_name = dict_to_str(sweep_dict)
        sub_dir = os.path.join(save_dir, subdir_name)

        if not os.path.exists(sub_dir):
            continue

        flattened_sweep = flatten_dict(sweep_dict)
        best_result_path = os.path.join(sub_dir, 'best_checkpoint')

        with open(best_result_path, 'r') as f:
            result_str = f.read()
            result_dict = eval(result_str)

        row = {key: flattened_sweep[map_column_to_full[key]] for key in columns}
        row[metric] = result_dict[metric]

        df = df.append(row, ignore_index=True)

    print(df)

