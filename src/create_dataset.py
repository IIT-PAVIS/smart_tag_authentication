import argparse
import logging
import os
from pathlib import Path

import cv2

import utils

import numpy as np

logging.basicConfig(level=logging.INFO)

# splits = {
#     "IMG_20210201_160905": "TEST",
#     "IMG_20210212_143250": "TEST",
#     "IMG_20210212_144031": "TEST",
#     "IMG_20210212_150057": "TEST",
#     "IMG_20210212_150548": "TEST",
#     "IMG_20210201_151027": "TEST",
#     "IMG_20210316_170002": "TRAIN",
#     "IMG_20210316_170514": "TRAIN",
#     "IMG_20210316_174334": "TRAIN",
#     "IMG_20210322_153713": "TRAIN",
#     "IMG_20210322_154022": "VAL",
#     "IMG_20210322_171300": "VAL",
#     "IMG_20210212_120014": "TEST",
#     "IMG_20210212_142159": "TEST",
#     "IMG_20210212_142819": "TEST",
#     "IMG_20210212_150834": "TEST",
#     "IMG_20210215_124420": "TEST",
#     "IMG_20210215_124815": "TEST",
# }


def analyze_image(image_filename, out_folder, splits, visualize=False):
    if visualize:
        cv2.namedWindow("HiQ Nano", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "HiQ Nano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    # Initializing the output filenames
    _, filename = os.path.split(image_filename)
    destination_dir = out_folder + splits[filename[:-4]] + "/" + filename[:-4] + "/"

    sub_images, sub_labels, image_unwarp, image_grid = utils.get_cells_from_images(
        image_filename
    )

    # Load the image
    image = cv2.imread(image_filename)

    if visualize:
        cv2.imshow(
            "HiQ Nano",
            utils.hconcat_resize_min(
                [
                    utils.hconcat_resize_min([image, image_grid]),
                    image_unwarp,
                ]
            ),
        )
        cv2.waitKey(0)
        cv2.destroyWindow("HiQ Nano")

    # Saving the images: creating all the required folders
    for label in range(4):
        Path(destination_dir + str(label)).mkdir(parents=True, exist_ok=True)

    for idx in range(len(sub_images)):
        label = sub_labels[idx]
        filename = "{}/{}/{:03d}.png".format(destination_dir, label, idx)

        cv2.imwrite(filename, sub_images[idx])


def main(args):
    in_folder = args.in_folder
    out_folder = args.out_folder

    # Looking for all jpg files in in_folder
    image_filenames = []
    for file in os.listdir(in_folder):
        if file.endswith(".jpg"):
            image_filenames.append(os.path.join(in_folder, file))
    
    # train test val splits
    splits = {}
    train_slice = int(.6 * len(image_filenames))
    val_slice = int(.8 * len(image_filenames))
    np.random.seed(42)
    shuffler = np.random.permutation(len(image_filenames))
    for filename in np.array(image_filenames)[shuffler][:train_slice]:
        _, filename = os.path.split(filename)
        splits[filename[:-4]] = 'TRAIN'
    for filename in np.array(image_filenames)[shuffler][train_slice:val_slice]:
        _, filename = os.path.split(filename)
        splits[filename[:-4]] = 'VAL'
    for filename in np.array(image_filenames)[shuffler][val_slice:]:
        _, filename = os.path.split(filename)
        splits[filename[:-4]] = 'TEST'


    # analyze
    for image_filename in image_filenames:
        logging.info(f"Analyzing {image_filename}")

        analyze_image(image_filename, out_folder, splits, args.visualize)


if __name__ == "__main__":
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_folder",
        type=str,
        default="./data/imgs/",
        help="input folder",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="./data/dataset/",
        help="output folder",
    )
    parser.add_argument(
        "--img_cols",
        type=int,
        default=15,
        help="img_cols",
    )
    parser.add_argument(
        "--img_rows",
        type=int,
        default=15,
        help="img_rows",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
    )

    options = parser.parse_args()

    main(options)
