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

import json
import multiprocessing
import os
from functools import partial
from urllib.request import Request, urlopen

import rofunc as rf
from rofunc.utils.oslab.path import get_rofunc_path


def download_ycb_objects(objects_to_download="all", files_to_download=['google_16k'],
                         extract=True, core_num=20):
    """
    Download YCB objects from the official website

    :param objects_to_download: List of objects to download. If "all", will download all objects. http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
    :param files_to_download: List of files to download for each object.
           'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
           'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
           'berkeley_processed' contains all of the segmented point clouds and textured meshes.
           'google_16k' contains google meshes with 16k vertices.
           'google_64k' contains google meshes with 64k vertices.
           'google_512k' contains google meshes with 512k vertices.
    :param extract: Extract all files from the downloaded .tgz, and remove .tgz files. If false, will just download all .tgz files to output_directory
    :param core_num: Number of cores to use for parallel downloading
    :return:
    """
    base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
    objects_url = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"

    # Define an output folder
    output_directory = os.path.join(get_rofunc_path(), "simulator/assets/urdf/ycb")
    rf.oslab.create_dir(output_directory)

    objects = fetch_objects(objects_url)

    pool = multiprocessing.Pool(core_num)
    parallel_1 = partial(parallel, objects_to_download=objects_to_download, files_to_download=files_to_download,
                         extract=extract, base_url=base_url, output_directory=output_directory)
    pool.map(parallel_1, objects)
    rf.logger.beauty_print("Downloaded all YCB objects to %s" % output_directory)


def parallel(object, objects_to_download, files_to_download, extract, base_url, output_directory):
    if objects_to_download == "all" or object in objects_to_download:
        for file_type in files_to_download:
            url = tgz_url(base_url, object, file_type)
            if not check_url(url):
                continue
            filename = "{path}/{object}_{file_type}.tgz".format(
                path=output_directory,
                object=object,
                file_type=file_type)
            download_file(url, filename)
            if extract:
                extract_tgz(filename, output_directory)


def fetch_objects(url):
    """ Fetches the object information before download """
    response = urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]


def download_file(url, filename, pbar=None):
    """ Downloads files from a given URL """
    u = urlopen(url)
    f = open(filename, "wb")
    file_size = int(u.getheader("Content-Length"))
    if pbar is not None:
        pbar.set_postfix_str("%s (%.2f MB)" % (os.path.basename(filename), file_size / 1000000.0))

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        # status = r"%10d  [%3.2f%%]" % (file_size_dl / 1000000.0, file_size_dl * 100. / file_size)
        # status = status + chr(8) * (len(status) + 1)
        # print(status)
    f.close()
    rf.logger.beauty_print("Downloaded %s" % filename)


def tgz_url(base_url, object, type):
    """ Get the TGZ file URL for a particular object and dataset type """
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object, type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object, type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object, type=type)


def extract_tgz(filename, dir):
    """ Extract a TGZ file """
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename, dir=dir)
    os.system(tar_command)
    os.remove(filename)


def check_url(url):
    """ Check the validity of a URL """
    try:
        request = Request(url)
        request.get_method = lambda: 'HEAD'
        response = urlopen(request)
        return True
    except Exception as e:
        return False


if __name__ == '__main__':
    download_ycb_objects()
