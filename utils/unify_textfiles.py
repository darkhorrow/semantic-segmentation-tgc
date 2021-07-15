#!/usr/bin/env python

import argparse
import glob
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dir_path',
                        help='Root path to the text files', required=True)

    parser.add_argument('-o', '--out_path',
                        help='Target text file path', required=True)

    parser.add_argument('-v', '--verbose', help='Display info about the progress', action='store_true')

    parser.add_argument('--exclude_names', nargs='+',
                        help='Text files names to be excluded (i.e README.txt, LICENSE.txt, etc)', default=[])

    args = parser.parse_args()

    filenames = [filename for filename in glob.glob(f'{args.dir_path}/**/*.txt', recursive=True)
                 if os.path.basename(filename) not in args.exclude_names]

    size = len(filenames)

    with open(args.out_path, 'w') as out:
        for index, filename in enumerate(filenames):
            n_lines = 0
            with open(filename, 'r') as file:
                for line in file.readlines():
                    out.write(line)
                    n_lines += 1

            if args.verbose:
                print(f'[{index + 1}/{size}] {filename}\nLines read: {n_lines}')
