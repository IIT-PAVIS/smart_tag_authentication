import os

import hydra
import pytorch_lightning as pl
from data_module import HiQNanoDataModule
from model import HiQNanoClassifier_ord_regr
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_name="./conf/config.yaml")
def main(cfg):


    print(OmegaConf.to_yaml(cfg))

    # Setting the seed
    if cfg.model.use_seed:
        pl.seed_everything(cfg.model.seed)

    dir_path = hydra.utils.get_original_cwd()

    # Initializing the data
    data_module = HiQNanoDataModule(
        data_augment_conf=cfg.trainer.data_augmentation,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        splits_dir=os.path.join(dir_path,cfg.model.input_splits_dir),
        data_dir=os.path.join(dir_path, cfg.model.input_data_dir),
        class_num=cfg.model.class_num,
        seed = cfg.model.seed,
    )
    data_module.setup()

    # Initializing the model
    model = HiQNanoClassifier_ord_regr(
        class_num=cfg.model.class_num,
        learning_rate=cfg.model.learning_rate,
        t_conv_size=cfg.model.t_conv_size,
        # weight=data_module.train_dataset.__get_class_balance__(),
        # use_pretrained=cfg.model.use_pretrained,
        # fix_layers=cfg.model.fix_layers,
    )

    # Initializing the saving callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        period=cfg.trainer.checkpoint_save_epoch_period,
        monitor="valid_acc",
        save_last=True,
    )

    # Starting the training
    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        gpus=cfg.trainer.gpus,
        deterministic=cfg.trainer.deterministic,
        auto_lr_find=cfg.trainer.auto_lr_find,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        max_epochs=cfg.trainer.max_epochs,
        auto_scale_batch_size=cfg.trainer.auto_scale_batch_size,
        benchmark=cfg.trainer.benchmark,
        profiler=cfg.trainer.profiler,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        resume_from_checkpoint=cfg.trainer.resume_from_checkpoint,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
