


'''

* Team_ID    :   HC#3344

* Filename   :   main.py

* Theme      :   E-yrc Homecoming



* Functions Used :

 parsing()

 predict(model,loader)

 animal_name(animal_image,path_animal)

 habitat_name(habitat_image,path_habitat)

'''

# Importing torch libraries

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

#importing argparse for argument parsing
import argparse


#Importing functions and class  from different module of package
from utils.dataset import  ImageDataset

from utils.class_names import habitat_class_name , animal_class_name




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(model,loader):
    '''

        Argumensts : model  : pytorch model for processing image
                     loader : pytorch dataloader of the Image

        Returns    : integer value of predicted value bt evaluating models

        Description :
                The image values are feed into model giving resultant value of identified class

    '''

    model.eval()
    for i, inputs in enumerate(loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)



            return int(preds)


def animal_name(animal_image,path_animal):

    '''

    Argumensts : animal_image : Path of image
                 path_animal  : Path of model

    Returns    : integer value of predicted value for the selection from class

    Description :

                Selecting the appropriate model for defining architecture and define the number of output classes.
                The fuction load a pretrained model and pass it to the predict function for output

    '''

    model=models.resnet18()

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 38)

    model = model.to(device)

    model.load_state_dict(torch.load(path_animal,map_location=device))

    loader = ImageDataset(animal_image)

    loader = val = torch.utils.data.DataLoader(loader,batch_size=1,
                                               shuffle=True)

    predicted=predict(model,loader)

    return int(predicted)





def habitat_name(habitat_image,path_habitat):
    '''

    Argumensts : habitat_image : Path of image
                 path_habitat  : Path of model

    Returns    : integer value of predicted value for the selection from class

    Description :

                Selecting the appropriate model for defining architecture and define the number of output classes.
                The fuction load a pretrained model and pass it to the predict function for output
    '''

    model=models.resnet18()

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 24)

    model =model.to(device)

    model.load_state_dict(torch.load(path_habitat,map_location=device))

    loader = ImageDataset(habitat_image)

    loader = val = torch.utils.data.DataLoader(loader,batch_size=1,
                                               shuffle=True)

    predicted=predict(model,loader)

    return int(predicted)

def parsing():


          '''


          Arguments: None

          Returns:
                     args.a,args.h,args.amod, args.hmod:

                      A tuple of 4 values consisting animal image , habitat image , animal model , habitat model



          Description:

                The function takes command Line argument and performs the following actions as they are mentioned in task details
          '''



          parser = argparse.ArgumentParser(add_help=False)

          #Animal Image
          parser.add_argument("-a",help="Input Animal Image",type=str)
          #Habitat Image
          parser.add_argument("-h",help="Input Habitat Image",type=str)
          #Animal model
          parser.add_argument("--amod",help="Load Animal model",type=str)
          #Habitat model
          parser.add_argument("--hmod",help="Load Habitat model",type=str)

          args = parser.parse_args()


          return args.a , args.h , args.amod , args.hmod



if __name__=="__main__":


    animal_image , habitat_image , animal_model , habitat_model = parsing()

    habiat_class = habitat_class_name

    animal_class = animal_class_name

    if animal_model != None:
        path_animal=animal_model
    path_animal = 'best_animal_resnet18.pth'

    if habitat_model != None:
        path_habitat = 'best_habitat_resnet18.pth'
    path_habitat = 'best_habitat_resnet18.pth'


    if animal_image != None:
        value=animal_name(animal_image,path_animal)
        animal_class = animal_class_name()
        print(animal_class[value])

    if habitat_image != None:
        value=habitat_name(habitat_image,path_habitat)
        habitat_class = habitat_class_name()
        print(habitat_class[value])
