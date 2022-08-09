import numpy as np
import pandas as pd
import os
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
import glob
import argparse
import warnings

#it ignores all warnings, in this example it is usefull cause MultiLabelBinarizer tells it doesn't see class ""
# and that is exactly what we need since we have exaples with no abnormalities
warnings.filterwarnings("ignore")

def main(path):
    all_xray_df = pd.read_csv('metadata/Data_Entry_2017.csv')


    all_image_paths = glob.glob(f'{path}/images_*/images/*.png', recursive=True)
    all_image_paths.sort()
    all_image_paths = {os.path.basename(x): x for x in all_image_paths}

    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('|', ','))

    classes = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split(',')).tolist())))
    classes = [x for x in classes if len(x)>0]
    classes.sort()

    encoder = MultiLabelBinarizer(classes=classes)
    labels = encoder.fit_transform([c.split(",") for c in list(all_xray_df["Finding Labels"])])

    df = pd.DataFrame()

    df["Image Index"] = all_xray_df["Image Index"]
    df["Label"] = labels.tolist()
    df["Path"] = all_xray_df["path"]

    train_val_list = pd.read_fwf('metadata/train_val_list.txt', header=None)
    train_val_list = train_val_list.squeeze()
    train_val_list.head()
    train_df = df.loc[all_xray_df['Image Index'].isin(train_val_list)]

    test_list = pd.read_fwf('metadata/test_list.txt', header=None)
    test_list = test_list.squeeze()
    test_df = df.loc[all_xray_df['Image Index'].isin(test_list)]
    test_df.head()

    print(f"{len(train_df)} examples in the trainset and {len(test_df)} examples in the testset, {len(train_df)+len(test_df)}  in total")


    train_df.to_csv("metadata/train_df.csv", index=False)
    test_df.to_csv("metadata/test_df.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="generate_official_split_test_train_dataframes.py",
        description=(
            "Generate train test dataframe with your custom paths to images"
            + ".\nExample command: \n"
            + " " * 4
            + "python generate_official_split_test_train_dataframes.py --path /home/piotr/Desktop/Nih-Chest-X-Ray-Artifacts-Annotations"
        ), formatter_class=argparse.RawTextHelpFormatter)


    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="path to Nih images directory",
    )

    args = parser.parse_args()
    path = args.path
    # path = "/home/piotr/Desktop/Nih-Chest-X-Ray-Artifacts-Annotations"

    main(path)
