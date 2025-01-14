import glob
import yaml
import pandas as pd
import numpy as np
import os
import argparse
import logging
from multiprocessing import Pool
from .util import makedir_if_not_exists, assert_files_exist, initialize_logging


def parse_immuneml_logistic_ml_details(ml_details_yaml):
    with open(ml_details_yaml, 'r') as f:
        ml_details = yaml.safe_load(f)
    feature_names = ml_details['feature_names']
    coefficients = np.array(ml_details['coefficients'], dtype=np.float32)
    return feature_names, coefficients


def create_kmer_index(feature_names):
    return {kmer: idx for idx, kmer in enumerate(feature_names)}

def kmer_encoder(sequence, k, kmer_to_index):
    counts = np.zeros(len(kmer_to_index), dtype=np.uint8)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_to_index:
            counts[kmer_to_index[kmer]] = 1
    return counts


def score_sequence(kmer_encoded_counts, weights):
    return np.dot(kmer_encoded_counts, weights)

def compute_score(sequence, k, kmer_to_index, weights):
    counts = kmer_encoder(sequence, k, kmer_to_index)
    score = score_sequence(counts, weights)
    return score

def compute_scores_for_all_sequences(sequences, feature_names, coefficients):
    kmer_to_index = create_kmer_index(feature_names)
    scores = []
    for sequence in sequences:
        scores.append(compute_score(sequence, 4, kmer_to_index, coefficients))
    return scores

def parse_simulated_dataset(simulated_dataset_path, feature_names, coefficients, metadata=None):
    if metadata is not None:
        metadata = pd.read_csv(metadata, header=0)
        true_class_examples = metadata[metadata['label_positive'] == True]
        true_class_examples = true_class_examples['filename'].to_list()
        pos_df = read_and_concatenate_dfs(simulated_dataset_path, true_class_examples, feature_names, coefficients)
        non_zero_scores = pos_df[pos_df['score'] != 0]
        non_zero_scores = non_zero_scores.sort_values(by='score', ascending=False)
        if non_zero_scores.shape[0] < 50000:
            top_50k = non_zero_scores
        else:
            top_50k = non_zero_scores.head(50000)
        top_50k_no_score = top_50k.drop(columns=['score'])
        return non_zero_scores, top_50k_no_score


def read_and_concatenate_dfs(simulated_dataset_path, examples_list, feature_names, coefficients):
    dfs_list = []
    for filename in examples_list:
        print("Processing file: ", filename)
        df = pd.read_csv(os.path.join(simulated_dataset_path, filename), sep='\t')
        scores = compute_scores_for_all_sequences(df['junction_aa'].to_list(), feature_names, coefficients)
        df['score'] = scores
        dfs_list.append(df)
    concatenated_df = pd.concat(dfs_list)
    return concatenated_df


def compute_overlap_with_true_signal(ml_details_yaml, true_signal_tsv, simulated_dataset_path, metadata_file,
                                     nonzero_scores_file, top50k_overlap_with_true_signal_file,
                                     jaccard_similarity_file):
    feature_names, coefficients = parse_immuneml_logistic_ml_details(ml_details_yaml)
    true_signal = parse_true_signal(true_signal_tsv)
    non_zero_scores, top_50k_no_score = parse_simulated_dataset(simulated_dataset_path, feature_names, coefficients,
                                                                metadata_file)
    non_zero_scores.to_csv(nonzero_scores_file, sep='\t', index=False)
    jaccard_similarity_df = compute_jaccard_similarity(top_50k_no_score, true_signal)
    jaccard_similarity_df.to_csv(jaccard_similarity_file, sep='\t', index=False)
    overlap = pd.merge(top_50k_no_score, true_signal, how='inner')
    overlap.to_csv(top50k_overlap_with_true_signal_file, sep='\t', index=False)


def parse_true_signal(true_signal_tsv):
    true_signal = pd.read_csv(true_signal_tsv, header=None, sep='\t', index_col=None)
    if true_signal.shape[1] > 3:
        true_signal.drop(true_signal.columns[0], axis=1, inplace=True)
    true_signal = true_signal.drop_duplicates()
    true_signal.columns = ['junction_aa', 'v_call', 'j_call']
    return true_signal


def compute_jaccard_similarity(predicted_sequences, true_sequences):
    predicted_sequences = predicted_sequences.drop_duplicates()
    true_sequences = true_sequences.drop_duplicates()
    n_true = true_sequences.shape[0]
    if predicted_sequences.shape[0] < n_true:
        dummy_df = pd.DataFrame({'junction_aa': ['dummy'] * (n_true - predicted_sequences.shape[0]),
                                 'v_call': ['dummy'] * (n_true - predicted_sequences.shape[0]),
                                 'j_call': ['dummy'] * (n_true - predicted_sequences.shape[0])})
        predicted_sequences = pd.concat([predicted_sequences, dummy_df])
    predicted_sequences = predicted_sequences.head(n_true)
    intersection = pd.merge(predicted_sequences, true_sequences, how='inner')
    union = pd.concat([predicted_sequences, true_sequences]).drop_duplicates()
    jaccard_index = intersection.shape[0] / union.shape[0]
    jaccard_similarity_df = pd.DataFrame({'jaccard_similarity': [jaccard_index]})
    return jaccard_similarity_df


def gather_config(super_path_sim_dirs, ml_out_super_path, root_path_to_replace):
    dir_list = glob.glob(f"{super_path_sim_dirs}/**/data", recursive=True)
    ml_output_dirs = [os.path.join(ml_out_super_path, base_dir.replace(root_path_to_replace, "").replace("/data", ""))
                      for base_dir in dir_list]
    interpretability_dirs = [os.path.join(base_dir, "interpretability") for base_dir in ml_output_dirs]
    ml_details_path = ("immuneml_output/hpoptim_instr/split_1"
                       "/label_positive_feature_size_4_logistic_regression_optimal/ml_details.yaml")
    true_signal_files = [os.path.join(base_dir, "signal_components", "filtered_implantable_signal_pool.tsv")
                         for base_dir in dir_list]
    simulated_datasets_paths = [os.path.join(base_dir, "train") for base_dir in dir_list]
    metadata_files = [os.path.join(base_dir, "train", "metadata.csv") for base_dir in dir_list]
    ml_details_yaml_files = [os.path.join(base_dir, ml_details_path) for base_dir in ml_output_dirs]
    for filelist in [true_signal_files, simulated_datasets_paths, metadata_files, ml_details_yaml_files]:
        assert_files_exist(filelist)
    for base_dir in interpretability_dirs:
        makedir_if_not_exists(base_dir)
    nonzero_scores_files = [os.path.join(base_dir, "nonzero_score_sequences.tsv") for base_dir in interpretability_dirs]
    top50k_overlap_with_true_signal_files = [os.path.join(base_dir, "top50k_seqs_overlapped_with_true_signal.tsv")
                                             for base_dir in interpretability_dirs]
    jaccard_similarity_files = [os.path.join(base_dir, "jaccard_similarity_with_true_signal.tsv") for base_dir in
                                interpretability_dirs]
    configs_list = []
    for ml_details_yaml, true_signal_tsv, simulated_dataset_path, metadata_file, nonzero_scores_file, \
            top50k_overlap_with_true_signal_file, jaccard_similarity_file in zip(ml_details_yaml_files,
                                                                                 true_signal_files,
                                                                                 simulated_datasets_paths,
                                                                                 metadata_files,
                                                                                 nonzero_scores_files,
                                                                                 top50k_overlap_with_true_signal_files,
                                                                                 jaccard_similarity_files):
        config = {'ml_details_yaml': ml_details_yaml, 'true_signal_tsv': true_signal_tsv,
                  'simulated_dataset_path': simulated_dataset_path, 'metadata_file': metadata_file,
                  'nonzero_scores_file': nonzero_scores_file,
                  'top50k_overlap_with_true_signal_file': top50k_overlap_with_true_signal_file,
                  'jaccard_similarity_file': jaccard_similarity_file}
        configs_list.append(config)
    return configs_list

def run_config(config):
    logging.info(f"Processing config: {config}")
    compute_overlap_with_true_signal(config['ml_details_yaml'], config['true_signal_tsv'],
                                     config['simulated_dataset_path'], config['metadata_file'],
                                     config['nonzero_scores_file'], config['top50k_overlap_with_true_signal_file'],
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


