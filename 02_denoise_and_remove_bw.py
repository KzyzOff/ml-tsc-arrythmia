import os
from multiprocessing import Pool, cpu_count
import numpy as np
from dotenv import load_dotenv
from scipy.io import loadmat, savemat
from external_tools.zheng_denoise import denoiser
from statsmodels.nonparametric.smoothers_lowess import lowess

from external_tools.utils import get_record_paths
from external_tools.zheng_denoise.denoiser import NLM_1dDarbon

load_dotenv()
DATASET_PATH = os.getenv('DATASET_PATH')
PREPROCESSED_DATASET_PATH = os.getenv('PREPROCESSED_DATASET_PATH')


def process_record(filepath: str):
    if not filepath.endswith('.mat'):
        return

    leadII = loadmat(DATASET_PATH + '/' + filepath)['val'][1, :]
    baseline = lowess(leadII, x, 0.01, return_sorted=False)

    filtered = leadII - baseline
    denoised_filtered = NLM_1dDarbon(filtered, 0.1, 5, 2)

    if not os.path.exists(PREPROCESSED_DATASET_PATH + '/' + filepath):
        savemat(PREPROCESSED_DATASET_PATH + '/' + filepath,
                {'leadII': denoised_filtered})
        print(f'File {filepath} done')
    else:
        print(f'Skipping {filepath}')


records = get_record_paths(DATASET_PATH + '/RECORDS')
record_paths = []
x = np.arange(5000)

for rec_p in records:
    os.makedirs(PREPROCESSED_DATASET_PATH + '/' + rec_p, exist_ok=True)

    for filename in os.listdir(DATASET_PATH + '/' + rec_p):
        record_paths.append(rec_p + filename)


if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        tasks = [path for path in record_paths]
        pool.map(process_record, tasks)
