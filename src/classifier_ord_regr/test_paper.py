import os
import sys

import albumentations as A
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import utils_3d_parties as ext_modules
from albumentations.pytorch import ToTensorV2
from data_module import HiQNanoDataModule
from dataset_class import HiQNanoDataset
from model import HiQNanoClassifier_ord_regr
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

sys.path.append("./src/")
import utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


@hydra.main(config_name="./conf/config.yaml")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    dir_path = hydra.utils.get_original_cwd()

    prediction_list = []
    gt_list = []

    # Initializing the models
    models = []

    models_path = cfg.model_pretrained.checkpoint_paths
    ensambling_n = len(models_path)
    for idx in range(ensambling_n):
        checkpoint = torch.load(
            os.path.join(dir_path, models_path[idx]),
            map_location=lambda storage, loc: storage,
        )

        model = HiQNanoClassifier_ord_regr(
            class_num=cfg.model.class_num,
            t_conv_size=cfg.model.t_conv_size,
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()



        models.append(model)

    transform_test = A.Compose(
        [
            A.RandomCrop(
                width=cfg.trainer.data_augmentation.image_size,
                height=cfg.trainer.data_augmentation.image_size,
            ),
            ToTensorV2(),
        ]
    )

            # Setting the seed
    if cfg.model.use_seed:
        pl.seed_everything(cfg.model.seed)

    # Loading test images (full images!)
    test_set_filename = os.path.join(dir_path, cfg.model.input_splits_dir) + "TEST_dataset.csv"
    df = pd.read_csv(test_set_filename)

    dir_names = df.dir_name.to_numpy()

    for dir_name in dir_names:
        norm_path = os.path.normpath(dir_name)
        path_components = norm_path.split(os.sep)
        img_filename = path_components[-1] + '.jpg'

        # img_path = os.path.join(dir_path, cfg.model.input_data_dir) + img_filename

        # Cropping the cells
        # sub_images, sub_labels, image_unwarp, image_gt = utils.get_cells_from_images(
        #     img_path
        # )

        list_imgs = []
        list_img_names = []
        list_classes = []
        for cl in os.listdir(dir_name):
            imgs = os.listdir(dir_name + cl)            
            list_img_names.extend(imgs)
            
            imgs = [f"{dir_name}{cl}/{img_name}" for img_name in imgs]
            classes = [cl for img_name in imgs]

            list_classes.extend(classes)
            list_imgs.extend(imgs)

        list_imgs = np.array(list_imgs)
        list_img_names = np.array(list_img_names)
        list_classes = np.array(list_classes)

        shuffler = np.random.permutation(len(list_img_names))
        list_imgs = list_imgs[shuffler]
        list_img_names = list_img_names[shuffler]
        list_classes = list_classes[shuffler]


        img_stack = None
        for img in list_imgs:  
            if os.path.isfile(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if transform_test is not None:
                    augmented = transform_test(image=img)
                    img = augmented["image"].float()

                img = np.expand_dims(img, 1)
            else:
                img = None
                print(img)

            if img_stack is None:
                img_stack = img
            else:
                img_stack = np.concatenate((img_stack, img), axis=1)

        img_stack = torch.from_numpy(np.expand_dims(img_stack, 0))
        # img_stack = np.expand_dims(img_stack, 0)

        # Classifying 
        logits_stack = None
        for idx in range(ensambling_n):
            logits = torch.sigmoid(models[idx](img_stack))
            logits = logits.view(-1, cfg.model.class_num - 1)
            logits = np.expand_dims(np.array(logits.detach().cpu()), 0)

            if logits_stack is None:
                logits_stack = logits
            else:
                logits_stack = np.concatenate((logits_stack, logits), axis=0)
        
        out = np.average(logits_stack, axis=0)
        # predictions = ext_modules.proba_to_label(torch.sigmoid(torch.from_numpy(out)))
        predictions = ext_modules.proba_to_label(torch.from_numpy(out))

        if cfg.model.class_num == 3:
            predicted += 1

        full_img = np.zeros((15*img_stack.shape[3], 15*img_stack.shape[4]))

        img_stack = img_stack.squeeze()
        img_stack = img_stack.permute(1, 2, 3, 0)

        # reorder for image reconstruction
        ind = np.argsort(list_img_names)
        predictions = predictions[ind]
        img_stack = img_stack[ind]
        list_classes = list_classes[ind]

        img_row = None
        img_full = None
        for idx in range(img_stack.shape[0]):
            if idx%15 == 0:
                if img_row is not None:
                    if img_full is None:
                        img_full = img_row
                    else:
                        img_full = np.concatenate((img_full, img_row), axis=0)
                
                img_row = None

            img = img_stack[idx].detach().cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = 255 * np.ones_like(img) ### modified to show annotations only

            pred = int(predictions[idx].detach().cpu())
            label = int(list_classes[idx])

            utils.add_label_pred(img, label, pred) ### comment out to show images only

            prediction_list.append(pred)
            gt_list.append(label)

            if img_row is None:
                img_row = img
            else:
                img_row = np.concatenate((img_row, img), axis=1)

        if img_full is None:
            img_full = img_row
        else:
            img_full = np.concatenate((img_full, img_row), axis=0)

        cv2.imwrite(f"Results_{img_filename}", img_full)

    cm = confusion_matrix(gt_list, prediction_list)
    acc = accuracy_score(gt_list, prediction_list)

    #TODO write into logs
    print("Confusion matrix")
    print(cm)
    print("Accuracy: {}".format(acc))
    
    from utils import cm_analysis
    cm_analysis(gt_list, prediction_list, 'confmat.jpg', [0,1,2,3], ymap=None, figsize=(10,10) )


if __name__ == "__main__":
    main()
