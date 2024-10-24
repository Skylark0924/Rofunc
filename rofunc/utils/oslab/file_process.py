#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

def delete_lines(in_path, out_path, head, tail=0):
    """
    Delete the appointed lines in the file.

    :param in_path: the input file path.
    :param out_path: the output file path.
    :param head: number of lines to be deleted from the head of the file.
    :param tail: number of lines to be deleted from the tail of the file.
    :return:
    """
    with open(in_path, 'r') as fin:
        a = fin.readlines()
    with open(out_path, 'w') as fout:
        b = ''.join(a[head:])
        fout.write(b)
