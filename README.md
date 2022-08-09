# Nih-Xray-Classifier

This repository contains [Pytorch Lighting](https://github.com/PyTorchLightning/pytorch-lightning) codebase for state-of-the-art image classifier for [NIH X-Ray Dataset](https://www.kaggle.com/nih-chest-xrays/data). 
We use the [EfficenNet](https://github.com/lukemelas/EfficientNet-PyTorch) as a model backbone and [Monai library](https://github.com/Project-MONAI/MONAI) for image transfomations. 


## Getting Started

You can start working with readme in three fast steps

1) Download the data

2) Generate dataframes with your custom paths

3) Start the training

Below we present further details regarding that three steps.

### Download the data

The dataset is avalible at [kaggle](https://www.kaggle.com/nih-chest-xrays/data). Unfortuately it is not posiible to download the dataset with a simple `wget` command with constant link, however you can use your cookie from Kaggle to do so. 

After downloading the data you should have the following file structure in the folder where you extracted the downloaded files:

```
├── images_001
│   └── images
├── images_002
│   └── images
├── ...
└── images_012
|    └── images
```
   

### Generate dataframes with your custom paths

To work properly the created Dataset (see [dataset.py](dataset.py)) needs a df with paths to images. In order to generate such run the [generate_official_split_test_train_dataframes.py](generate_official_split_test_train_dataframes.py).
You can use the the below command

```
python generate_official_split_test_train_dataframes.py --path /path/to/nih/images
```

After script execution you should see, if you don't something didn't work (check again the given path)

```
Scans found: 112120 , Total Headers 112120
86524 examples in the trainset and 25596 examples in the testset, 112120  in total
```

After that you should have files `test_df.csv` and `train_df.csv` in the `metadata` folder. Both csv files should have such structure:

|Image Index|Label|Path|
| :-------------: |:-------------:|:-------------:|
|00000001_000.png|"[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"|/home/user/Nih-Images/images_001/images/00000001_000.png
|00000001_001.png|"[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]"|/home/user/Nih-Images/images_001/images/00000001_001.png
|00000001_003.png|"[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]"|/home/user/Nih-Images/images_001/images/00000001_002.png

If you look for more details on what actually happen in the script, check out the [Generate official split test train dataframes notebook](Generate-official-split-test-train-dataframes.ipynb) where we descirbe each step in detail. 

### Start the training

The whole training process takes place in the [training.py](training.py) file. In you want to start trainign with defaut configuration just run

```
python training.py
```

If you want to change any training parameterss you have to do it manually in [code](training.py).

```python
logger = TensorBoardLogger(TENSORBOARD_DIRECTORY, name="efficentnet_fine_tuning")

model = EfficennetLightning(batch_size=1024)

trainer = Trainer(max_epochs=10,
                  logger=logger,
                  gpus=1,
                  accumulate_grad_batches=4,
                  deterministic=True,
                  early_stop_callback=True)

trainer.fit(model)
```
## Results

Our model achieves state-of-the-art results for the task of image classification. 


## Authors

* [Piotr Mazurek](https://github.com/tugot17)
* [Kajetan Dymkiewicz](https://github.com/Drakles/)
* [Szymon Adamski](https://github.com/Neutriv)
* [Jakub Kufel](https://github.com/Gucio926)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
