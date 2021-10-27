# HiQNano
Supporting code for the paper "Nanocatalyst-enabled physically unclonable functions as smart anticounterfeiting tags with AI-aided smartphone authentication".

## Installation
```
pip install pipenv
pipenv install
```

## Dataset creation
The provided data are expected to be in `filename.jpg` format and placed in `./data/imgs/` folder with the related annotation file, named `filename.csv`.

The current dataset can be provided upon request. All the contained images should should be unzipped into `./data/imgs/`.

Annotate the 4 code corners and reorganize the data using the following command:
```
 pipenv shell
 python3 ./src/create_dataset.py
```

It will create a `./dataset/` folder and `TRAIN`, `VAL` and `TEST` subfolders with the annotated images organized as follow:

    ./dataset/
        filename_1/
            0/
            1/
            2/
            3/
        filename_2/
            0/
            1/
            2/
            3/
        ...
        filename_N/
            0/
            1/
            2/
            3/

PAY ATTENTION THAT CURRENTLY THE SPLITS ARE HARD-CODED IN create_dataset.py !!!!

## Model Training
The model implementation is in the `classifier_ord_regr` folder. 

The model has been implemented in `model.py`, while the dataloaders have been implemented in `data_module.py`.

## Classifier Ordinal Regression
### Create Data Split
Edit `./src/classifier_ord_regr/conf/conf_datasplit.yaml` for creating TRAIN, VAL and TEST data splits. Then, launch:
```
pipenv shell
python3 ./src/classifier_ord_regr/create_data_splits.py
```

Training can be launched using:
```
python3 ./src/classifier_ord_regr/train.py
```

