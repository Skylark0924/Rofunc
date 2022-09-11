from pbdlib.gui import InteractiveDemos, MutliCsInteractiveDemos
import argparse

"""
Utilities to record demonstrations

Press "h" for help once in the GUI.

On left is the position, on right the velocity. dt is 0.05 by default.

"""

arg_fmt = argparse.RawDescriptionHelpFormatter

parser = argparse.ArgumentParser(formatter_class=arg_fmt)

parser.add_argument(
	'-f', '--filename', dest='filename', type=str,
	default='test', help='filename for saving the demos'
)
parser.add_argument(
	'-p', '--path', dest='path', type=str,
	default='', help='path for saving the demos'
)

parser.add_argument(
	'-m', '--multi_cs', dest='multi_cs', action='store_true',
	default=False, help='record demos in multiple coordinate systems'
)
parser.add_argument(
	'-c', '--cs', dest='nb_cs', type=int,
	default=2, help='number of coordinate systems'
)

args = parser.parse_args()

if args.multi_cs:
	interactive_demo = MutliCsInteractiveDemos(
		filename=args.filename, path=args.path, nb_experts=args.nb_cs)
else:
	interactive_demo = InteractiveDemos(
		filename=args.filename, path=args.path)

interactive_demo.start()