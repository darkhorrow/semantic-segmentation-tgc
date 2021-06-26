#!/usr/bin/env python

"""
The purpose of this script is to convert the YOLO annotation format
o a custom format to feed a faster RCNN

The YOLO format is a JSON file with entries as this example:

{
 "frame_id":2527,
 "filename":"/content/drive/MyDrive/TFM/TGCRBNWv0.1UnifiedImgs/TGC_2525.jpg",
 "objects": [
  {"class_id":2, "name":"car", "relative_coordinates":{"center_x":0.102191, "center_y":0.384652, "width":0.067044, "height":0.043052}, "confidence":0.960477},
  {"class_id":2, "name":"car", "relative_coordinates":{"center_x":0.985373, "center_y":0.374936, "width":0.028488, "height":0.033985}, "confidence":0.884116},
  {"class_id":2, "name":"car", "relative_coordinates":{"center_x":0.951051, "center_y":0.375410, "width":0.040345, "height":0.033010}, "confidence":0.802839},
  {"class_id":0, "name":"person", "relative_coordinates":{"center_x":0.242514, "center_y":0.427185, "width":0.036019, "height":0.177366}, "confidence":0.986528}
 ]
}

Keep in mind that the images needs to be accessible in order to work,
since YOLO does not save image dimensions information.
"""
import pandas as pd
import argparse
import cv2

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

    parser.add_argument('-i', '--json_path',
                        help='Root path to the JSON with the annotations', required=True)

    parser.add_argument('-o', '--out_path',
                        help='Target path to store formatted annotations', required=True)

    parser.add_argument('-v', '--verbose', help='Display info about the progress', action='store_true')

    args = parser.parse_args()

    input_df = pd.read_json(args.json_path)

    output_df = pd.DataFrame(columns=OUTPUT_COLS)

    row = 0

    for index, (path, objects_detected) in enumerate(zip(input_df['filename'], input_df['objects'][:])):
        for element, object_detected in enumerate(objects_detected):
            if object_detected['name'] == 'person':
                img = cv2.imread(path)
                dh, dw, _ = img.shape

                x = object_detected['relative_coordinates']['center_x']
                y = object_detected['relative_coordinates']['center_y']
                w = object_detected['relative_coordinates']['width']
                h = object_detected['relative_coordinates']['height']

                l = int((x - w / 2) * dw)
                r = int((x + w / 2) * dw)
                t = int((y - h / 2) * dh)
                b = int((y + h / 2) * dh)

                if l < 0:
                    l = 0
                if r > dw - 1:
                    r = dw - 1
                if t < 0:
                    t = 0
                if b > dh - 1:
                    b = dh - 1

                output_df.loc[row, 'class'] = object_detected['name']
                output_df.loc[row, 'filename'] = path
                output_df.loc[row, 'x'] = l
                output_df.loc[row, 'y'] = t
                output_df.loc[row, 'x2'] = r
                output_df.loc[row, 'y2'] = b

                if args.verbose:
                    print(f'[{row}] {path}')

                row = row + 1

    output_df.to_csv(args.out_path, index=False)
