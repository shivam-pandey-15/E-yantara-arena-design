

'''

* Team_ID    :   HC#3344

* Filename   :   load_data.py

* Theme      :   E-yrc Homecoming



* Functions Used :

 load_data()



'''
from utils.dataset import  create_and_load_meta_csv_df , ImageDataset

import torch
import os
import torch.utils.data

def load_data():
    '''

        Argumensts : None

        Returns    : Tuple of 4 values containg animal loader habitat loader animal DataFrame habitat DataFrame respectively

        Description :
                    The function loads the data from dataset file
    '''

    dataset_path=destination_path='Images/'
    animal_df,habitat_df=create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True)


    animal_loader = ImageDataset(animal_df)

    habitat_loader = ImageDataset(habitat_df)

    animal_predict =torch.utils.data.DataLoader(animal_loader,batch_size=20,
                                                   shuffle=False)

    habitat_predict = torch.utils.data.DataLoader(habitat_loader,batch_size=25,
                                                   shuffle=False)




    return animal_predict , habitat_predict , animal_df , habitat_df
