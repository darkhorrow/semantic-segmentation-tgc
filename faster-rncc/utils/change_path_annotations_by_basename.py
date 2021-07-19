#!/usr/bin/env python

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dir_path',
                        help='Root path to the images in the desired path', required=True)

    parser.add_argument('-f', '--file_path',
                        help='Annotation file path to edit', required=True)

    parser.add_argument('--extensions', nargs='+',
                        help='Image format [jpg, png, gif, jpeg, bmp]', default=['jpg', 'png', 'gif', 'jpeg', 'bmp'])

    args = parser.parse_args()

    index = 0
    previous = None
    new_lines = []

    with open(args.file_path, 'r') as out:
        for line in out.readlines():
            line_parts = line.split(',')
            file_name = os.path.basename(line_parts[0])
            new_file_path = os.path.join(args.dir_path, file_name)
            line_parts[0] = new_file_path
            new_lines.append(','.join(line_parts))

    with open(args.file_path, 'w') as out:
        out.writelines(new_lines)
