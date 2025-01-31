import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


def get_record_paths(path: str) -> list:
    record_paths = []
    with open(path, 'r') as records_file:
        for x in records_file:
            record_paths.append(x.rstrip('\n'))

    return record_paths


def load_data(dataset_path: os.PathLike | str) -> tuple[np.ndarray, np.ndarray]:
    load_dotenv()

    labels_df = pd.read_csv(
        os.path.join(dataset_path, 'labels.csv'),
        sep=',',
        usecols=[1],
        names=['filename', 'label']
    )
    y = labels_df['label'].tolist()
    print('Loaded labels')

    paths = []
    with open(os.path.join(dataset_path, 'RECORDS'), 'r') as records_file:
        for x in records_file:
            paths.append(x.rstrip('\n'))

    X = []

    for path in paths:
        for file in os.listdir(os.path.join(dataset_path, path)):
            mat_file = loadmat(
                os.path.join(dataset_path, path, file)
            )['leadII'].flatten()

            X.append(mat_file)

    print('Loaded features')

    return np.array(X), np.array(y)


def data2npz(record_path: os.PathLike | str):
    load_dotenv()
    dataset_path = os.getenv('DATASET_PATH')

    labels_df = pd.read_csv(
        'data/labels.csv',
        sep=',',
        usecols=[1],
        names=['filename', 'label']
    )
    labels = labels_df['label'].tolist()

    record_paths = []
    for rec_p in get_record_paths(record_path):
        for filename in os.listdir(dataset_path + '/' + rec_p):
            record_paths.append(rec_p + filename)

    ecg_signals = [loadmat(dataset_path + '/' + filepath)['val'][1, :]
                   for filepath in record_paths]

    np.savez_compressed('data/database_1_raw.npz',
                        label=np.array(labels),
                        ecg=ecg_signals)


def apply_butter(signal, sampling_rate=500, cutoff_freq=0.5):
    nyq_freq = 0.5 * sampling_rate
    cutoff_norm = cutoff_freq / nyq_freq
    b, a = butter(4, cutoff_norm, btype='high', analog=False)

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


if __name__ == '__main__':
    data2npz('data/database_1/RECORDS')
