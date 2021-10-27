import logging
import os
from pathlib import Path
from shutil import copy2

import hydra
import pandas as pd
from omegaconf import OmegaConf
from sklearn.utils import shuffle


def check_splits_common_elements(split_1_csv, split_2_csv):
    dataset_1 = pd.read_csv(
        split_1_csv,
        skiprows=1,
        names=["idx", "name", "label"],
    )["name"].to_numpy()

    dataset_2 = pd.read_csv(
        split_2_csv,
        skiprows=1,
        names=["idx", "name", "label"],
    )["name"].to_numpy()

    dataset_1_as_set = set(dataset_1)

    intersection = dataset_1_as_set.intersection(dataset_2)

    if len(intersection) > 0:
        print("!!! Data splits have elements in common !!!!")

    return len(intersection) > 0


def create_df_datasplit(img_folder):
    splits = {}

    list_subfolders = os.listdir(img_folder)

    for subfolder in list_subfolders:
        list_images = os.listdir(f"{img_folder}/{subfolder}/")
        list_images = [
            f"{img_folder}/{subfolder}/{img_name}/" for img_name in list_images
        ]
        df_list_images = pd.DataFrame(list_images, columns=["dir_name"])
        df_list_images = shuffle(df_list_images)

        splits[subfolder] = df_list_images

    return splits


@hydra.main(config_name="./conf/conf_datasplit.yaml")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    dir_path = hydra.utils.get_original_cwd()

    img_folder = os.path.join(dir_path, cfg.img_folder)
    out_folder = os.path.join(dir_path, cfg.out_splits_dir)

    # Splitting available images in Train, Val, Test based on the img subfolder organization
    dataset_imgs_splits = create_df_datasplit(img_folder)

    for key in dataset_imgs_splits.keys():
        logging.info(f"  {key}: {len(dataset_imgs_splits[key])} images.")

    # Saving the generated splits and configuration file
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for key in dataset_imgs_splits.keys():
        dataset_imgs_splits[key].to_csv(out_folder + f"/{key}_dataset.csv")

    copy2(
        dir_path + "/src/classifier_ord_regr/conf/conf_datasplit.yaml",
        out_folder,
    )

    # Verify no elements are in common between the splits
    for key1 in dataset_imgs_splits.keys():
        for key2 in dataset_imgs_splits.keys():
            if key1 != key2:
                _ = check_splits_common_elements(
                    out_folder + f"/{key1}_dataset.csv",
                    out_folder + f"/{key2}_dataset.csv",
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
