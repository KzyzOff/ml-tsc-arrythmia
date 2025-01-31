import os
from dotenv import load_dotenv

from utils import get_record_paths


def assign_label(dx_str: str):
    dx_list = dx_str.split(',')
    if '426783006' in dx_list and len(dx_list) == 1:
        return 0
    else:
        return 1


load_dotenv()
DATASET_PATH = os.getenv('DATASET_PATH')
LABELS_FILE = os.getenv('DENOISED_DATASET_PATH') + '/labels.csv'

record_paths = get_record_paths(DATASET_PATH + '/RECORDS')

with open(LABELS_FILE, 'a') as out_file:
    for base_path in record_paths:
        for file in os.listdir(DATASET_PATH + '/' + base_path):
            if not file.endswith('.hea'):
                continue

            with open(DATASET_PATH + '/' + base_path + '/' + file, 'r') as hea_file:
                lines = hea_file.readlines()

                dx_line = next((line for line in lines if line.startswith('#Dx:')), None)
                if dx_line:
                    dx_content = dx_line.split(':')[1].strip()

                    label = assign_label(dx_content)

            out_file.write(f'{os.path.basename(file)},{label}\n')

        print(f'File {base_path} done')
