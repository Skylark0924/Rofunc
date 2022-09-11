#!/usr/bin/env python
from pbdlib.gui import InteractiveDemos
import argparse

print("dt = 0.05")

arg_fmt = argparse.RawDescriptionHelpFormatter

parser = argparse.ArgumentParser(formatter_class=arg_fmt)

parser.add_argument(
	'-f', '--filename', dest='filename', type=str,
	default='test', help='filename'
)
parser.add_argument(
	'-p', '--path', dest='path', type=str,
	default='', help='path'
)
args = parser.parse_args()

interactive_demo = InteractiveDemos(filename=args.filename, path=args.path)
interactive_demo.start()
