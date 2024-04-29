# Based on https://github.com/Nordgaren/Github-Folder-Downloader

import os
from github import Github, Repository, ContentFile
import requests
from argparse import ArgumentParser, Namespace


def download(c: ContentFile, out: str, is_file: bool):
    r = requests.get(c.download_url)
    if is_file:
        output_path = out
    else:
        output_path = os.path.join(out, os.path.basename(c.path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        print(f"downloading {c.path} to {out}")
        f.write(r.content)


def download_folder(repo: Repository, folder: str, out: str, recursive: bool):
    contents = repo.get_contents(folder)
    for c in contents:
        if c.download_url is None:
            if recursive:
                download_folder(repo, c.path, out, recursive)
            continue
        download(c, out, False)


def download_file(repo: Repository, file_path: str, out: str):
    c = repo.get_contents(file_path)
    download(c, out, True)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--repo", default="", help="The repo where the file or folder is stored")
    parser.add_argument("--path", default="", help="The folder or file you want to download")
    parser.add_argument(
        "-o",
        "--out",
        default="downloads",
        required=False,
        help="Path to folder you want to download "
        "to. Default is current folder + "
        "'downloads'",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively download directories. Folder " "downloads, only!",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        help="Set flag to download a single file, instead of a " "folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.repo = "NVIDIA-Omniverse/IsaacGymEnvs"
    args.path = "assets/urdf/ycb/025_mug"
    g = Github()
    repo = g.get_repo(args.repo)
    if args.file:
        download_file(repo, args.path, args.out)
    else:
        download_folder(repo, args.path, args.out, args.recursive)
