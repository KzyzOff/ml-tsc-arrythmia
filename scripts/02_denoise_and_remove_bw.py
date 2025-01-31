import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.io import loadmat
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils import get_record_paths, apply_butter
from denoiser import NLM_1dDarbon

load_dotenv()
DATASET_PATH = '../' + os.getenv('DATASET_PATH')


def process_record(filepath: str) -> np.array:
    leadII = loadmat(DATASET_PATH + '/' + filepath)['val'][1, :]
    sig_norm = (leadII - np.min(leadII)) / (np.max(leadII) - np.min(leadII))
    sig_butter = apply_butter(sig_norm)

    sig_denoised = NLM_1dDarbon(sig_butter, 0.1, 5, 2)
    baseline = lowess(sig_denoised, np.linspace(0, 10, len(sig_denoised)), 0.01, return_sorted=False)

    print(f'File {filepath} processed')

    return sig_denoised - baseline


if __name__ == '__main__':
    records = get_record_paths(DATASET_PATH + '/RECORDS')
    labels_df = pd.read_csv(
        '../data/labels.csv',
        sep=',',
        usecols=[1],
        names=['filename', 'label']
    )
    y = np.array(labels_df['label'].tolist())

    record_paths = []
    for rec_p in records:
        for filename in os.listdir(DATASET_PATH + '/' + rec_p):
            if filename.endswith('.mat'):
                record_paths.append(rec_p + filename)

    with Pool(processes=cpu_count() - 2) as pool:
        processed_signals = pool.map(process_record, record_paths)

    print('Saving to .npz file...')
    np.savez_compressed('../data/database_1_processed.npz',
                        labels=y,
                        ecg=processed_signals)
