import argparse
import copy
import glob
import os
import multiprocessing
from time import sleep
from .util import parse_user_yaml, put_values_in_target_key, makedir_if_not_exists, write_yaml_file, \
    assert_files_exist, divide_into_sublists


def generate_ml_configs(super_path_sim_dirs, ml_yaml_file, output_dir, method_name, root_path_to_replace):
    dir_list = glob.glob(f"{super_path_sim_dirs}/**/data", recursive=True)
    sim_dir_list = [os.path.join(base_dir, "simulated_repertoires") for base_dir in dir_list]
    metadata_list = [os.path.join(simdata_path, "metadata.csv") for simdata_path in sim_dir_list]
    train_metadata_list = [os.path.join(base_dir, "train", "metadata.csv") for base_dir in dir_list]
    test_metadata_list = [os.path.join(base_dir, "test_metadata.csv") for base_dir in dir_list]
    ml_config = parse_user_yaml(ml_yaml_file)
    ml_configs_list = [put_values_in_target_key(copy.deepcopy(ml_config), "path", simdata_path) for simdata_path in
                       sim_dir_list]
    mod_ml_configs = [put_values_in_target_key(ml_config_dict, "metadata_file", meta_path)
                      for ml_config_dict, meta_path in zip(ml_configs_list, metadata_list)]
    mod_ml_configs = [put_values_in_target_key(ml_config_dict, "train_metadata_path", train_meta_path) for
                      ml_config_dict, train_meta_path in zip(mod_ml_configs, train_metadata_list)]
    mod_ml_configs = [put_values_in_target_key(ml_config_dict, "test_metadata_path", test_meta_path) for
                      ml_config_dict, test_meta_path in zip(mod_ml_configs, test_metadata_list)]
    assert_files_exist(metadata_list)
    assert_files_exist(train_metadata_list)
    assert_files_exist(test_metadata_list)
    ml_out_paths = [
        os.path.join(output_dir, f"{method_name}_ml_output",
                     path.replace(root_path_to_replace, "").replace("/data", "")) for
        path in dir_list]
    ml_config_out_fns = [os.path.join(ml_out_path, "ml_config_file.yaml") for ml_out_path in ml_out_paths]
    for outpath in ml_out_paths:
        makedir_if_not_exists(outpath)
    for mod_ml_config, file_path in zip(mod_ml_configs, ml_config_out_fns):
        write_yaml_file(mod_ml_config, file_path)
    return ml_config_out_fns, ml_out_paths


def run_jobs(n_parallel_jobs, delay_minutes, ml_config_out_fns, ml_out_paths):
    parallel_jobs_list = divide_into_sublists(list(zip(ml_config_out_fns, ml_out_paths)), n_parallel_jobs)
    with multiprocessing.Pool(n_parallel_jobs) as pool:
        processes = []
        for parallel_jobs in parallel_jobs_list:
            for ml_config, output_folder in parallel_jobs:
                p = pool.apply_async(run_immuneml, (ml_config, os.path.join(output_folder, "immuneml_output")))
                processes.append(p)
                sleep(delay_minutes * 60)
        for p in processes:
            p.get()


def run_immuneml(yaml_file, output_folder):
    command = f'immune-ml {yaml_file} {output_folder}'
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running immune-ml failed:{command}.")


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--super_path_sim_dirs', help='path to directory that contains all simulated datasets',
                        required=True)
    parser.add_argument('-m', '--ml_yaml_file', help='ML config file for immune-ml', required=True)
    parser.add_argument('-l', '--ml_method_name', help='Output directory for ML results', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory for ML results', required=True)
    parser.add_argument('-n', '--n_parallel_jobs', help='Number of parallel jobs to run', required=True, type=int)
    parser.add_argument('-d', '--delay_minutes', help='Delay in minutes between jobs', required=True, type=int)
    parser.add_argument('-r', '--root_path_to_replace', help='Root path to replace in ml output paths',
                        required=True)
    args = parser.parse_args()
    ml_config_out_fns, ml_out_paths = generate_ml_configs(args.super_path_sim_dirs, args.ml_yaml_file,
                                                          args.output_dir, args.ml_method_name,
                                                          args.root_path_to_replace)
    run_jobs(args.n_parallel_jobs, args.delay_minutes, ml_config_out_fns, ml_out_paths)
