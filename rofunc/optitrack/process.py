import os

import pandas as pd
from tqdm import tqdm


def delete_lines(in_path, out_path, head, tail=0):
    with open(in_path, 'r') as fin:
        a = fin.readlines()
    with open(out_path, 'w') as fout:
        b = ''.join(a[head:])
        fout.write(b)


def data_clean(input_path):
    demo_csvs = os.listdir(input_path)
    demo_csvs = sorted(demo_csvs)
    for demo_csv in demo_csvs:
        if 'Take' in demo_csv:
            if 'Manus' in demo_csv:
                out_path = os.path.join(input_path, 'process')
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                demo_path = os.path.join(input_path, demo_csv)
                out_file_path = os.path.join(out_path, demo_csv)
                delete_lines(demo_path, out_file_path, 14)
                csv_data = pd.read_csv(out_file_path)
                # csv_data = pd.read_csv(demo_path, skiprows=6)
                if '3f6ec26f' in demo_csv:
                    csv_data.to_csv(os.path.join(input_path, "left_manus.csv"))
                elif '7b28f20b' in demo_csv:
                    csv_data.to_csv(os.path.join(input_path, "right_manus.csv"))
            else:
                demo_path = os.path.join(input_path, demo_csv)
                # The first 6 rows are headers: https://v22.wiki.optitrack.com/index.php?title=Data_Export:_CSV
                csv_data = pd.read_csv(demo_path, skiprows=6)
                csv_data.to_csv(os.path.join(input_path, "opti_hands.csv"))
    print('{} finished'.format(input_path.split('/')[-1]))


def data_clean_batch(input_dir):
    demos = os.listdir(input_dir)
    demos = sorted(demos)
    for demo in tqdm(demos):
        input_path = os.path.join(input_dir, demo)
        data_clean(input_path)
