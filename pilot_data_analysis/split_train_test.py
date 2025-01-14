import argparse
import glob
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from util import makedir_if_not_exists

TRAIN_DIR = 'train'
TEST_DIR = 'test'

def split_data(df, test_size=0.5, balanced_test=False):
    y = df['label_positive']
    if balanced_test:
        positive = df[df['label_positive']]
        negative = df[~df['label_positive']]
        num_test_per_class = int(len(df) * test_size / 2)
        test_positive = positive.sample(n=num_test_per_class)
        test_negative = negative.sample(n=num_test_per_class)
        test_df = pd.concat([test_positive, test_negative])
        test = test_df.sample(frac=1).reset_index(drop=True)
        train_df = df.drop(test_df.index)
        train = train_df.sample(frac=1).reset_index(drop=True)
    else:
        train, test = train_test_split(df, test_size=test_size, stratify=y)
    return train, test

def create_dirs_and_move_files(train_df, test_df, original_dir):
    train_dir = os.path.join(original_dir, TRAIN_DIR)
    test_dir = os.path.join(original_dir, TEST_DIR)
    makedir_if_not_exists(train_dir)
    makedir_if_not_exists(test_dir)
    copy_from_source_to_destination(original_dir, train_df, train_dir)
    copy_from_source_to_destination(original_dir, test_df, test_dir)
    write_metadata_to_csv(test_df, os.path.join(original_dir, "test_metadata.csv"))
    write_metadata_to_csv(train_df, os.path.join(train_dir, "metadata.csv"))


def write_metadata_to_csv(metadata_df, metadata_path):
    metadata_df.to_csv(metadata_path, index=False)

def copy_from_source_to_destination(source_dir, metadata_df, destination_dir):
    for _, row in metadata_df.iterrows():
        source = os.path.join(source_dir, "simulated_repertoires", row['filename'])
        dest = os.path.join(destination_dir, row['filename'])
        shutil.copy2(source, dest)

def main(original_dir, test_size=0.5, balanced_test=False):
    df = pd.read_csv(os.path.join(original_dir, "simulated_repertoires", 'metadata.csv'), usecols=['subject_id',
                                                                                                   'filename',
                                                                                                   'label_positive'])
    train, test = split_data(df, test_size=test_size, balanced_test=balanced_test)
    create_dirs_and_move_files(train, test, original_dir)

def process_multiple_dirs(super_path):
    dir_list = glob.glob(os.path.join(super_path, "**/data"), recursive=True)
    for original_dir in dir_list:
        train_dir_path = os.path.join(original_dir, TRAIN_DIR)
        test_dir_path = os.path.join(original_dir, TEST_DIR)
        test_metadata_path = os.path.join(original_dir, "test_metadata.csv")
        if os.path.exists(train_dir_path) and os.path.exists(test_dir_path) and os.path.exists(test_metadata_path):
            print(f"Splitting has been already done for {original_dir}, skipping...")
            continue
        else:
            main(original_dir)

def execute_on_multiple_dirs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--super_path', help='path to directory that contains all simulated datasets',
                        required=True)
    args = parser.parse_args()
    process_multiple_dirs(args.super_path)

def execute_on_single_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='path to directory that needs train and test split',
                        required=True)
    parser.add_argument('-t', '--test_size', help='proportion of total examples to be in test set', type=float,
                        required=False, default=0.5)
    parser.add_argument('-b', '--balanced_test', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to balance the test set irrespective of the balance in training set')
    args = parser.parse_args()
    main(args.dir, args.test_size, args.balanced_test)
