#!/usr/bin/env python

"""
Split a train/test partition of files and its annotations files
"""
import pandas as pd
import numpy as np
import argparse
import os
import shutil

CSV_HEADER = [
    'filename',
    'x',
    'y',
    'x2',
    'y2',
    'class'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images_path',
                        help='Root path to the images', required=True)

    parser.add_argument('-a', '--annotate_path',
                        help='Root path to the annotations', required=True)

    parser.add_argument('-r', '--train_path',
                        help='Target path to store train images/annotations', required=True)

    parser.add_argument('-e', '--test_path',
                        help='Target path to store test images/annotations', required=True)

    parser.add_argument('-s', '--train_size',
                        help='Train size value between 0.0 and 1.0. The rest will be used for test', type=float,
                        choices=np.arange(0, 1.01, 0.01), metavar="[0.0-1.0]", required=True)

    parser.add_argument('-v', '--verbose', help='Display info about the progress', action='store_true')

    args = parser.parse_args()

    annotations_df = pd.read_csv(args.annotate_path, names=CSV_HEADER)

    train_df = pd.DataFrame(columns=CSV_HEADER)
    test_df = pd.DataFrame(columns=CSV_HEADER)

    images = os.listdir(args.images_path)
    np.random.shuffle(images)

    train_images, test_images = np.split(np.array(images), [int(len(images) * args.train_size)])

    if args.verbose:
        print(f'Train size: {len(train_images)}\tTest size: {len(test_images)}')
        print('Copying train images to target folder...')

    for image in train_images:
        train_df = train_df.append(annotations_df[annotations_df['filename'] == image], ignore_index=True)
        shutil.copy(os.path.join(args.images_path, image), os.path.join(args.train_path, image))
        if args.verbose:
            print(f'{os.path.join(args.images_path, image)} --> {os.path.join(args.train_path, image)}')

    if args.verbose:
        print('Copying test images to target folder...')

    for image in test_images:
        test_df = test_df.append(annotations_df[annotations_df['filename'] == image], ignore_index=True)
        shutil.copy(os.path.join(args.images_path, image), os.path.join(args.test_path, image))
        if args.verbose:
            print(f'{os.path.join(args.images_path, image)} --> {os.path.join(args.test_path, image)}')

    if args.verbose:
        print(f'Train annotations: {len(train_df.index)}\tTest annotations: {len(test_df.index)}')

    train_df.to_csv(os.path.join(args.train_path, 'annotateTrain.csv'), header=False, index=False)
    test_df.to_csv(os.path.join(args.test_path, 'annotateTest.csv'), header=False, index=False)
