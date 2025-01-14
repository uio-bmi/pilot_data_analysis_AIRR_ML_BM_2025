import argparse
import glob
import logging
import os
from multiprocessing import Pool
import pandas as pd
from fisher import pvalue_npy
import numpy as np
from util import assert_files_exist, makedir_if_not_exists, initialize_logging
from pilot_data_analysis.logistic_interpretability import parse_true_signal, compute_jaccard_similarity


def parse_simulated_dataset(simulated_dataset_path, metadata):
    metadata = pd.read_csv(metadata, header=0)
    true_class_examples = metadata[metadata['label_positive'] == True]
    false_class_examples = metadata[metadata['label_positive'] == False]
    true_class_examples = true_class_examples['filename'].to_list()
    false_class_examples = false_class_examples['filename'].to_list()
    pos_df = read_and_concatenate_dfs(simulated_dataset_path, true_class_examples)
    neg_df = read_and_concatenate_dfs(simulated_dataset_path, false_class_examples)
    pos_df['label_positive'] = 1
    neg_df['label_positive'] = 0
    concatenated_df = pd.concat([pos_df, neg_df])
    return concatenated_df


def read_and_concatenate_dfs(simulated_dataset_path, examples_list):
    dfs_list = []
    for filename in examples_list:
        df = pd.read_csv(os.path.join(simulated_dataset_path, filename), sep='\t')
        df['filename'] = filename
        dfs_list.append(df)
    concatenated_df = pd.concat(dfs_list)
    return concatenated_df


def count_unique_occurrences_of_rows(df):
    return df.groupby(['junction_aa', 'v_call', 'j_call', 'label_positive'], as_index=False).size()


def fisher_exact_test(df, pos_rep_size, neg_rep_size):
    df['pos_not_present'] = pos_rep_size - df['pos_count']
    df['neg_not_present'] = neg_rep_size - df['neg_count']
    contingency_npy = df[['pos_count', 'pos_not_present', 'neg_count', 'neg_not_present']].values
    contingency_npy = contingency_npy.astype(dtype=np.uint)
    _, right_side_p_value, twosided = pvalue_npy(contingency_npy[:, 0], contingency_npy[:, 1],
                                                 contingency_npy[:, 2], contingency_npy[:, 3])
    contin_odd_npy = contingency_npy + 1
    odds = (contin_odd_npy[:, 0] * contin_odd_npy[:, 3]) / (contin_odd_npy[:, 1] * contin_odd_npy[:, 2])
    df['odds_ratio'] = odds
    df['p_value'] = twosided
    df['right_side_p_value'] = right_side_p_value
    return df


def score_instances_on_training_datasset(training_dataset_path, training_metadata_path):
    concatenated_df = parse_simulated_dataset(training_dataset_path, training_metadata_path)
    neg_rep_size = concatenated_df[concatenated_df['label_positive'] == 0]['filename'].nunique()
    pos_rep_size = concatenated_df[concatenated_df['label_positive'] == 1]['filename'].nunique()
    counts_df = count_unique_occurrences_of_rows(concatenated_df)
    result = counts_df.pivot_table(index=['junction_aa', 'v_call', 'j_call'],
                                   columns='label_positive',
                                   values='size',
                                   fill_value=0).reset_index()
    result.columns.name = None
    result.rename(columns={1: "pos_count", 0: "neg_count"}, inplace=True)
    fisher_df = fisher_exact_test(result, pos_rep_size, neg_rep_size)
    fisher_df_top_50k = fisher_df.sort_values(by='right_side_p_value', ascending=True).head(50000)
    fisher_df_top_50k = fisher_df_top_50k[['junction_aa', 'v_call', 'j_call']]
    fisher_df_top_50k = fisher_df_top_50k.drop_duplicates()
    return fisher_df_top_50k


def compute_overlap_with_true_signal(training_dataset_path, training_metadata_path, true_signal_tsv,
                                     predicted_signal_file, top50k_overlap_with_true_signal_file,
                                     jaccard_similarity_file):
    predicted_signal = score_instances_on_training_datasset(training_dataset_path, training_metadata_path)
    predicted_signal.to_csv(predicted_signal_file, sep='\t', index=False)
    true_signal = parse_true_signal(true_signal_tsv)
    jaccard_similarity_df = compute_jaccard_similarity(predicted_signal, true_signal)
    jaccard_similarity_df.to_csv(jaccard_similarity_file, sep='\t', index=False)
    overlap = pd.merge(predicted_signal, true_signal, how='inner')
    overlap.to_csv(top50k_overlap_with_true_signal_file, sep='\t', index=False)


def gather_config(super_path_sim_dirs, ml_out_super_path, root_path_to_replace):
    dir_list = glob.glob(f"{super_path_sim_dirs}/**/data", recursive=True)
    training_dataset_paths = [os.path.join(base_dir, "train") for base_dir in dir_list]
    training_metadata_paths = [os.path.join(base_dir, "train", "metadata.csv") for base_dir in dir_list]
    true_signal_files = [os.path.join(base_dir, "signal_components", "filtered_implantable_signal_pool.tsv")
                         for base_dir in dir_list]
    ml_output_dirs = [os.path.join(ml_out_super_path, base_dir.replace(root_path_to_replace, "").replace("/data", ""))
                      for base_dir in dir_list]
    interpretability_dirs = [os.path.join(base_dir, "interpretability") for base_dir in ml_output_dirs]
    for base_dir in interpretability_dirs:
        makedir_if_not_exists(base_dir)
    for filelist in [true_signal_files, training_dataset_paths, training_metadata_paths, interpretability_dirs]:
        assert_files_exist(filelist)
    predicted_signal_files = [os.path.join(base_dir, "predicted_sequences_top_50k.tsv") for base_dir in
                              interpretability_dirs]
    top50k_overlap_with_true_signal_files = [os.path.join(base_dir, "top50k_seqs_overlapped_with_true_signal.tsv")
                                             for base_dir in interpretability_dirs]
    jaccard_similarity_files = [os.path.join(base_dir, "jaccard_similarity_with_true_signal.tsv") for base_dir in
                                interpretability_dirs]
    configs_list = []
    for training_dataset_path, training_metadata_path, true_signal_tsv, predicted_signal_file, \
            top50k_overlap_with_true_signal_file, jaccard_similarity_file in zip(training_dataset_paths,
                                                                                 training_metadata_paths,
                                                                                 true_signal_files,
                                                                                 predicted_signal_files,
                                                                                 top50k_overlap_with_true_signal_files,
                                                                                 jaccard_similarity_files):
        config = {'training_dataset_path': training_dataset_path, 'training_metadata_path': training_metadata_path,
                  'true_signal_tsv': true_signal_tsv, 'predicted_signal_file': predicted_signal_file,
                  'top50k_overlap_with_true_signal_file': top50k_overlap_with_true_signal_file,
                  'jaccard_similarity_file': jaccard_similarity_file}
        configs_list.append(config)
    return configs_list

def run_config(config):
    logging.info(f"Processing config: {config}")
    compute_overlap_with_true_signal(config['training_dataset_path'], config['training_metadata_path'],
                                     config['true_signal_tsv'], config['predicted_signal_file'],
                                     config['top50k_overlap_with_true_signal_file'],
                                     config['jaccard_similarity_file'])

def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--super_path_sim_dirs', help='path to directory that contains all simulated datasets',
                        required=True)
    parser.add_argument('-m', '--ml_out_super_path', help='path to directory that contains all ml output directories',
                        required=True)
    parser.add_argument('-n', '--n_threads', help='number of jobs to run in parallel', type=int,
                        required=True)
    parser.add_argument('-r', '--root_path_to_replace', help='root path to replace in ml output paths',
                        required=True)
    args = parser.parse_args()
    configs_list = gather_config(args.super_path_sim_dirs, args.ml_out_super_path, args.root_path_to_replace)
    log_fn = os.path.join(args.ml_out_super_path, "interpretability_computation_log.txt")
    initialize_logging(log_file_path=log_fn)
    pool = Pool(args.n_threads)
    pool.map(run_config, configs_list)
