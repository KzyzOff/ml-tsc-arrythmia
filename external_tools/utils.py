import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.io import loadmat


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
