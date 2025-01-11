import os
from dotenv import load_dotenv

from external_tools.utils import get_record_paths

load_dotenv()
DATASET_PATH = os.getenv('DATASET_PATH')
LABELS_FILE = os.getenv('PREPROCESSED_DATASET_PATH') + '/labels.csv'

record_paths = get_record_paths(DATASET_PATH + '/RECORDS')
# with open(DATASET_PATH + '/RECORDS', 'r') as records_file:
#     for x in records_file:
#         record_paths.append(x.rstrip('\n'))

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

                    label = 0 if dx_content.lower() == 'unknown' else 1

                else:
                    label = 0

            out_file.write(f'{os.path.basename(file)},{label}\n')

        print(f'File {base_path} done')
