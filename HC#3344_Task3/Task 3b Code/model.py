


'''


* Team_ID    :   HC#3344

* Filename   :   class_names.py

* Theme      :   E-yrc Homecoming


Functions Used:


predict(model,loader,class_names,df)

animal_prediction(animal_predict, animal_class , animal_df,path)

habitat_prediction(habitat_predict,habitat_class,habitat_df,path)



'''







import torch
import torch.nn as nn

from torchvision import models


from utils.load_data import load_data

from class_names import animal_class_name , habitat_class_name


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(model,loader,class_names,df):
    '''

        Argumensts : model  : pytorch model for processing image
                     loader : pytorch dataloader of the Image
                     class_names : python dictionary of value to the output
                     df : pandas dataframe containing image details


        Returns    :
                    python dictionay of image position and prediction

        Description :
                The image values are feed into model giving resultant value of identified classes

    '''

    answer={}
    images_so_far=0
    model.eval()
    for i, (inputs,labels) in enumerate(loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


            for j in range(inputs.size()[0]):
                answer[df.iloc[images_so_far]['name']] = class_names[int(preds[j])]

                images_so_far += 1

    return answer



def animal_prediction(animal_predict, animal_class , animal_df,path):

    '''

    Argumensts : animal_predict : loader of animal
                 animal_class   : Dictionary of animal_class
                 animal_df      : Animal dataframe
                 path  : Path of model

    Returns    : Dictionary of Animal Position with name

    Description :

                Selecting the appropriate model for defining architecture and define the number of output classes.
                The fuction load a pretrained model and pass it to the predict function for output

    '''

    model = models.resnet18()

    model.fc=nn.Linear(512,38)

    model.load_state_dict(torch.load(path,map_location=device))

    model = model.to(device)

    animal=predict(model,animal_predict,animal_class,animal_df)

    return animal

def habitat_prediction(habitat_predict,habitat_class,habitat_df,path):


    '''

        Argumensts : habitat_predict : loader of habitat
                     habitat_class   : Dictionary of habitat_class
                     habitat_df      : Habitat dataframe
                     path  : Path of model

        Returns    : Dictionary of Habitat Position with name

        Description :

                    Selecting the appropriate model for defining architecture and define the number of output classes.
                    The fuction load a pretrained model and pass it to the predict function for output

    '''

    model = models.resnet18()

    model.fc=nn.Linear(512,24)

    model.load_state_dict(torch.load(path,map_location=device))

    model = model.to(device)

    habitat=predict(model,habitat_predict,habitat_class,habitat_df)

    return habitat

def evaluate(amod,hmod):


    '''

        Argumensts : amod  : pytorch model for processing image
                     loader : pytorch dataloader of the Image

        Returns    : a tuple of dictionaries containg the output value

        Description :
                The image values are feed into model giving resultant value of identified class

    '''


    animal_predict , habitat_predict , animal_df , habitat_df = load_data()

    animal=animal_prediction( animal_predict , animal_class_name() , animal_df , amod)

    habitat=habitat_prediction( habitat_predict , habitat_class_name() , habitat_df, hmod)

    return animal , habitat
