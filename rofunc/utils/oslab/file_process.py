# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
