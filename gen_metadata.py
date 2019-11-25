"""
============================
Audio metadata extraction - for Zalo AI challenge
============================
Author: Le Anh Tho
"""
import numpy as np
import pandas as pd
# import scipy
import os
from tqdm import tqdm
import argparse
import glob
import multiprocessing
from utils.audio_metadata import *
import csv

if __name__ == '__main__':
	# Get script args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_dir', help='directory of the audios', default='./audio-data/train')
    parser.add_argument(
        '--output_dir', help='output directory', default='./audio-features')
    parser.add_argument('--n_cores', type=int, help='number of cores',
                        default=multiprocessing.cpu_count())
    parser.add_argument('--output_name', help='name of the output csv',
                        default='train_song_metadata.csv')

    args = parser.parse_args()
    path = args.audio_dir
    output_dir = args.output_dir
    n_cores = args.n_cores
    output_name = args.output_name

    print("Audio path: {}".format(path))
    print("Using {} cores in parallel".format(n_cores))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Create the list of files 
    all_files = glob.glob(os.path.join(path, "*.mp3"))
    # print(all_files[:10])
    # print(len(all_files))
    dfs = []
    pool = multiprocessing.Pool(n_cores)

    # Process the files in parallel
    results = pool.imap_unordered(get_metadata, all_files)
    for ft in tqdm(results, total=len(all_files)):
        dfs.append(ft)

    output = pd.concat(dfs, ignore_index=True)
    # Write the output to csv
    with open(os.path.join(output_dir, output_name), 'w') as csv_file:
        field_names = ['id', 'album', 'genre', 'duration']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in output.itertuples():
            # print(row[0], row[1])
            writer.writerow({
                'id': row[1],
                'album': row[2],
                'genre': row[3],
                'duration': row[4]
            })
