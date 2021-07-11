#!/usr/bin/env python

import argparse
import glob
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dir_path',
                        help='Root path to rename files', required=True)

    parser.add_argument('-o', '--out_path',
                        help='Target path to store new images', required=True)

    parser.add_argument('-p', '--prefix',
                        help='Prefix in the name of the images. If empty, original image name is used instead')

    parser.add_argument('-v', '--verbose', help='Display info about the progress', action='store_true')

    parser.add_argument('--extensions', nargs='+',
                        help='Image format [jpg, png, gif, jpeg, bmp]', default=['jpg', 'png', 'gif', 'jpeg', 'bmp'])

    args = parser.parse_args()

    filenames = [filename for extension in args.extensions for filename in
                 glob.glob(f'{args.dir_path}/**/*.{extension}', recursive=True)]

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    digits = f'0{len(str(len(filenames)))}d'

    for index, filename in enumerate(filenames):
        _, extension = os.path.splitext(filename)
        new_filename = f'{args.prefix}_{index:{digits}}{extension}' \
            if args.prefix is not None else os.path.basename(filename)
        new_filepath = os.path.join(args.out_path, new_filename)

        shutil.copy(filename, new_filepath)
        if args.verbose:
            print(f'[{index + 1}/{len(filenames)}] {filename} --> {new_filepath}')
