#!/usr/bin/env python

import argparse
import glob
import os
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dir_path',
                        help='Root path to images', required=True)

    parser.add_argument('-s', '--size',
                        help='Images with height/width under this parameter will be deleted', default=64, type=int)

    parser.add_argument('-v', '--verbose', help='Display info about the progress', action='store_true')

    parser.add_argument('--extensions', nargs='+',
                        help='Image format [jpg, png, gif, jpeg, bmp]', default=['jpg', 'png', 'gif', 'jpeg', 'bmp'])

    args = parser.parse_args()

    filenames = [filename for extension in args.extensions for filename in
                 glob.glob(f'{args.dir_path}/**/*.{extension}', recursive=True)]

    count = {
        'deleted': 0,
        'kept': 0
    }

    for index, filename in enumerate(filenames):
        im = cv2.imread(filename)
        h, w, _ = im.shape

        removed = False

        if h < int(args.size) or w < w < int(args.size):
            os.remove(filename)
            removed = True

        if removed:
            count['deleted'] = count['deleted'] + 1
        else:
            count['kept'] = count['kept'] + 1

        if args.verbose:
            print(f'[{index + 1}/{len(filenames)}] {filename} --> {"Removed" if removed else "Kept"} (w:{w}, h:{h})')

    if args.verbose:
        print(f'Files kept: {count["kept"]}\tFiles removed: {count["deleted"]}\t'
              f'Total images: {count["kept"] + count["deleted"]}')