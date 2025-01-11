import os
from dotenv import load_dotenv


def get_record_paths(path: str) -> list:
    load_dotenv()

    record_paths = []
    with open(path, 'r') as records_file:
        for x in records_file:
            record_paths.append(x.rstrip('\n'))

    return record_paths
