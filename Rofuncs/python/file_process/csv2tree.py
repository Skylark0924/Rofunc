import csv
import os
import argparse


def csv2tree(file_path):
    if not os.path.exists(file_path):
        print('Please type the correct file path!')
        return True

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        head = reader[0]

        for row in reader[1:]:
            for col_index in range(1, len(row)):
                name = head[col_index]
                val = row[col_index]
                print('{} {};'.format(name, val))
            print('/n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path of the csv file')
    args = parser.parse_args()

    csv2tree(args.path)
