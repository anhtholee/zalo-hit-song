"""
============================
Model ensembling - for Zalo AI challenge
============================
Author: Le Anh Tho
"""
# Import libraries
import numpy as np
import pandas as pd
# import xgboost as xgb
from tqdm import tqdm
from scipy.stats import hmean
from scipy.stats.mstats import gmean
# from utils.df_preprocessing import *
# from utils.models import NaiveRankEstimator
import glob, os
import sys

if __name__ == '__main__':

    # ==============================
    # ==== LOAD PREDICTION DATA ====
    # ==============================
    if not os.path.exists("submissions"):
        os.mkdir("submissions")
    submissions = glob.glob('ensemble-models/*.csv')
    preds = pd.read_csv('data/test_info.tsv', delimiter='\t', header=0, names=['id'], usecols=[0])
    submissions = sorted(submissions)
    colnames = []
    for i,f in enumerate(submissions):
        colname = os.path.basename(f).split('.')[0]
        temp = pd.read_csv(f, header=None, usecols=[1])
        preds[colname] = temp.iloc[:,0].values
        colnames.append(colname)
    # print(preds.head())
    ws = [0.23847736, 0.17293857, 0.00106586, 0.04447066, 0.44001623] # loss 1.570766 - xgb + cb + naive + nn + xgb(embedding)
    normalised_ws = np.round(np.array(ws) / np.sum(ws), 4)
    preds['label'] = np.clip(np.average(preds[colnames].to_numpy(), axis=1, weights=normalised_ws).round(4), 1, 10)
    print("Creating CSV file...")
    final = preds[['id', 'label']].to_numpy(dtype='float')
    np.savetxt('submissions/final_submission.csv', final, delimiter=',', fmt='%i,%1.4f')