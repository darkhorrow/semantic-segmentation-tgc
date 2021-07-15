#!/usr/bin/env python

import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dir_path',
                        help='Root path to the images in the desired path', required=True)

    parser.add_argument('-f', '--file_path',
                        help='Annotation file path to edit', required=True)

    parser.add_argument('--extensions', nargs='+',
                        help='Image format [jpg, png, gif, jpeg, bmp]', default=['jpg', 'png', 'gif', 'jpeg', 'bmp'])

    args = parser.parse_args()

    filenames = [filename for extension in args.extensions for filename in
                 glob.glob(f'{args.dir_path}/**/*.{extension}', recursive=True)]

    index = 0
    previous = None
    new_lines = []

    with open(args.file_path, 'r') as out:
        for line in out.readlines():
            line_parts = line.split(',')
            aux = line_parts[0]

            if aux != previous:
                index += 1

            line_parts[0] = filenames[index - 1]
            new_lines.append(','.join(line_parts))

            previous = aux

    with open(args.file_path, 'w') as out:
        out.writelines(new_lines)
