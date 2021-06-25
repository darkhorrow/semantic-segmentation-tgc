#!/usr/bin/env python

"""
The purpose of this script is to convert the CSV annotation format
given by the labeling tool https://www.makesense.ai/ to a custom format
to feed a faster RCNN

The CSV format has the following content per entry

<class> <x_upper_left_point> <y_upper_left_point> <width> <height> <filename> <image_width> <image_height>
"""
import pandas as pd
import argparse

CSV_HEADER = [
    'class',
    'x',
    'y',
    'width',
    'height',
    'filename',
    'image_width',
    'image_height'
]

OUTPUT_COLS = [
    'filename',
    'x',
    'y',
    'x2',
    'y2',
    'class'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--csv_path',
                        help='Root path to the CSV with the annotations', required=True)

    parser.add_argument('-o', '--out_path',
                        help='Target path to store formatted annotations', required=True)

    parser.add_argument('-v', '--verbose', help='Display info about the progress', action='store_true')

    args = parser.parse_args()

    input_df = pd.read_csv(args.csv_path, names=CSV_HEADER)

    output_df = pd.DataFrame(columns=OUTPUT_COLS)
    output_df['filename'] = input_df['filename']
    output_df['class'] = input_df['class']
    output_df['x'] = input_df['x']
    output_df['y'] = input_df['y']
    output_df['x2'] = input_df['width'] + input_df['x']
    output_df['y2'] = input_df['height'] + input_df['y']

    output_df.to_csv(args.out_path, header=False, index=False)
