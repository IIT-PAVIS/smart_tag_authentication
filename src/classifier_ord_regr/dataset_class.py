import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
import cv2


class HiQNanoDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_num, el_to_keep=-1, transform=None, seed=None):

        self.el_to_keep = el_to_keep
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.class_num = class_num

        self.dataset = pd.read_csv(csv_file)

        if seed is not None:
            torch.manual_seed(seed)
        # if self.class_num == 2:
        #     self.dataset.loc[self.dataset["Label"] != 0, "Label"] = 1
        # el

        # self.image_filenames = np.asarray(self.dataset["Image Name"])
        # self.labels = np.asarray(self.dataset["Label"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # np.random.seed()
        img_dir = self.dataset.iloc[idx].dir_name

        # Images of each class are contained in a different folder
        list_labels = []
        list_imgs = []
        for cl in os.listdir(img_dir):
            imgs = os.listdir(img_dir + cl)
            imgs = [f"{img_dir}{cl}/{img_name}" for img_name in imgs]
            list_imgs.extend(imgs)
            list_labels.extend([int(cl)] * len(imgs))

        list_imgs = np.array(list_imgs)
        list_labels = np.array(list_labels)

        if self.class_num == 2:
            list_labels[list_labels != 0] = 1
        if self.class_num == 3:
            list_imgs = list_imgs[list_labels != 0]
            list_labels = list_labels[list_labels != 0] - 1

        # Shuffling images and labels accordingly
        # shuffler = np.random.permutation(len(list_labels))
        shuffler = torch.randperm(len(list_labels))
        # print(shuffler[:5])
        list_imgs = list_imgs[shuffler]
        list_labels = list_labels[shuffler]

        # Piping all the images in a big stack
        if self.el_to_keep > 0:
            list_imgs = list_imgs[: self.el_to_keep]
            list_labels = list_labels[: self.el_to_keep]

        img_stack = None
        for img in list_imgs:  # filename = self.image_filenames[idx]
            if os.path.isfile(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.transform is not None:
                    augmented = self.transform(image=img)
                    img = augmented["image"].float()

                img = np.expand_dims(img, 1)
            else:
                img = None
                print(img)

            if img_stack is None:
                img_stack = img
            else:
                img_stack = np.concatenate((img_stack, img), axis=1)

        list_labels = list_labels.astype("int")
        ## MBMB - Deal with different class_n cases

        return (img_stack, list_labels, img_dir)

    def __len__(self):
        return len(self.dataset)

    def __get_class_balance__(self):
        labels_cnt = {}

        labels = np.unique(self.dataset["Label"].to_numpy())
        for label in labels:
            cnt = len(self.dataset["Label"][self.dataset["Label"] == label].to_numpy())

            labels_cnt[label] = cnt

        # # Normalizing the values # Not needed!!
        # factor = 1.0 / sum(labels_cnt.values())
        # for k in labels_cnt:
        #     labels_cnt[k] = labels_cnt[k] * factor

        data = list(labels_cnt.items())
        np_labels_cnt = np.array(data)[:, 1:].squeeze()

        if self.class_num == 2:
            labels_cnt_2 = []
            labels_cnt_2.append(np_labels_cnt[0])
            labels_cnt_2.append(sum(np_labels_cnt[1:]))
            np_labels_cnt = np.array(labels_cnt_2)

        return sum(np_labels_cnt) - np_labels_cnt
