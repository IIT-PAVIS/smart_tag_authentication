import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from dataset_class import HiQNanoDataset
from torch.utils.data import DataLoader


class HiQNanoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_augment_conf,
        class_num,
        splits_dir="./",
        data_dir="./",
        batch_size=64,
        num_workers=8,
        seed=None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.class_num = class_num

        self.train_set_filename = splits_dir + "TRAIN_dataset.csv"
        self.valid_set_filename = splits_dir + "VAL_dataset.csv"
        self.test_set_filename = splits_dir + "TEST_dataset.csv"
        self.data_dir = data_dir

        self.data_augment_conf = data_augment_conf

        self.seed = seed

    def setup(self, stage=None):
        transform_train = A.Compose(
            [
                A.ShiftScaleRotate(
                    p=self.data_augment_conf.train.shift_scale_rotate.p,
                    rotate_limit=self.data_augment_conf.train.shift_scale_rotate.rotate_limit,
                    shift_limit=self.data_augment_conf.train.shift_scale_rotate.shift_limit,
                    scale_limit=self.data_augment_conf.train.shift_scale_rotate.scale_limit,
                ),
                A.RandomCrop(
                    width=self.data_augment_conf.image_size,
                    height=self.data_augment_conf.image_size,
                ),
                A.HorizontalFlip(
                    p=self.data_augment_conf.train.horizontal_flip.p,
                ),
                A.RandomBrightnessContrast(
                    self.data_augment_conf.train.random_brightness_contrast.p,
                ),
                A.ToGray(
                    p=self.data_augment_conf.train.to_gray.p,
                ),
                ToTensorV2(),
            ]
        )

        transform_test = A.Compose(
            [
                A.RandomCrop(
                    width=self.data_augment_conf.image_size,
                    height=self.data_augment_conf.image_size,
                ),
                ToTensorV2(),
            ]
        )

        self.train_dataset = HiQNanoDataset(
            csv_file=self.train_set_filename,
            root_dir=self.data_dir,
            transform=transform_train,
            class_num=self.class_num,
            seed = self.seed,
        )
        self.validation_dataset = HiQNanoDataset(
            csv_file=self.valid_set_filename,
            root_dir=self.data_dir,
            transform=transform_test,
            class_num=self.class_num,
            seed = self.seed,
        )
        self.test_dataset = HiQNanoDataset(
            csv_file=self.test_set_filename,
            root_dir=self.data_dir,
            transform=transform_test,
            class_num=self.class_num,
            seed = self.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
