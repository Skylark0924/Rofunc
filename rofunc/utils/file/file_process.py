import pathlib
import rofunc as rf


def delete_lines(in_path, out_path, head, tail=0):
    with open(in_path, 'r') as fin:
        a = fin.readlines()
    with open(out_path, 'w') as fout:
        b = ''.join(a[head:])
        fout.write(b)


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    rf.utils.beauty_print('{} not exist, created.'.format(path), type='info')

