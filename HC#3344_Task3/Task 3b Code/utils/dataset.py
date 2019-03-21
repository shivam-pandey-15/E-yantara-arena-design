


'''

* Team_ID    :   HC#3344

* Filename   : dataset.py

* Theme      :   E-yrc Homecoming



'''







# import required libraries
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import torchvision


class_names={}

def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.

    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    temp_dir=os.getcwd()
    dataset_path=os.chdir(dataset_path)
    j=-1

    for i in os.listdir():
        if os.path.isdir(i):

            j+=1
            os.chdir(i)
            class_names[j]=i
            os.chdir('..')

    if 'dataset_attr.csv' not in os.listdir():

        # Make a csv with full file path and labels
        d=[]

        j=-1
        for i in os.listdir():
            j+=1

            os.chdir(i)
            class_names[j]=i
            for img in os.listdir():

                temp=os.getcwd()+'\\'+img,j,os.path.splitext(img)[0]
                a=Image.open(temp[0])
                a=np.array(a)
                if len(a.shape)==3 and a.shape[2]==3:
                    d.append(temp)
            os.chdir('..')
        df=pd.DataFrame(d,columns=['path','label','name'])

        # change destination_path to DATASET_PATH if destination_path is None
        if destination_path == None:
            destination_path = os.getcwd()

        # write out as dataset_attr.csv in destination_path directory
        df.to_csv('dataset_attr.csv')

        # if no error
    os.chdir(temp_dir)

    return True

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a
    fraction in split parameter.

    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """

    if create_meta_csv(dataset_path, destination_path=destination_path):
        os.chdir(dataset_path)
        dframe = pd.read_csv(os.getcwd()+'\\dataset_attr.csv')
        os.chdir('..')
    # shuffle if randomize is True or if split specified and randomize is not specified
    # so default behavior is split
    animal_df  = dframe[dframe.label==0]
    habitat_df = dframe[dframe.label==1]

    return animal_df,habitat_df



class ImageDataset(Dataset):
    """Image Dataset that works with images

    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.

    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform =torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        self.classes = data['label'].unique()# get unique classes from data dataframe


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        image = Image.open(img_path)# load PIL image

        label = self.data.iloc[idx]['label']# get label (derived from self.classes; type: int/long) of image

        if self.transform:
            image = self.transform(image)

        return image, label
