


'''




* Team_ID    :   HC#3344

* Filename   :   generate_image.py

* Theme      :   E-yrc Homecoming


Functions Used:

    find_animal(contours, hierarchy)

    find_contours(read_input_image)

    find_habitat(contours, hierarchy)

    generate_animal(image, animal, contours, hierarchy)

    generate_habitat(image, habitat, box)

    process_image(image, save)

    read_image(image)

    save_contour(image, animal, habitat, contours, hierarchy, box)


Description:
    Predefined classes for outputs.

'''



import cv2
import numpy as np
from copy import deepcopy
import os
import shutil
import argparse

def read_image(image):

    '''
        Argument:

        name: Name of input Image

        Return:

        Resized Input image, sahpe of orignal image.


        Description:


        The functions takes the image name as argument and convert the image intp an numpy array

        It converts the image into 800*800 since we are dealing with object of different shape and we are doing the naming by using points location of pixels whose values are retrived by marking locations in Paint.



    '''


    read_input_image  = cv2.imread(image,1)
    return read_input_image


def find_contours (read_input_image):

    '''
        Argument:

        read_input_image : numpy array of image

        Return:

        A tuple of 3 values consist of image contours location and hierarchy


        Description:


        The function takes image as input and find contours



    '''

    imgray = cv2.cvtColor(read_input_image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,240,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return image, contours, hierarchy


def find_animal(contours , hierarchy):

    '''
            Argument:

            contours  : A list consisting of the pixel value of each contour in an image.

            hierarchy : A numpy array of single element consiting a numpy array og length of contours each having 4 sub index,

            having value of Next , Previous , Child , Parent contours.


            Return:

            animal : A similar type data as in contours but only having the contours inside which there is an animal


            Description:


            It finds the outter box contour having the animal


    '''


    total_list = [i for i in range ( len (contours)) if hierarchy[0][i][3] == 0]


    external_box_child = {i: [] for i in range(len(contours)) }

    for i in range(len(contours)):

        if hierarchy[0][i][3] in external_box_child.keys():

            external_box_child[hierarchy[0][i][3]].append(i)


    for i in external_box_child.keys() :

            for j in range(len(external_box_child[i])):

                external_box_child[i][j]=cv2.contourArea(contours[external_box_child[i][j]])

            if len(external_box_child[i]) >0:

                external_box_child[i]=max(external_box_child[i])

            else :

                external_box_child[i]=0


    animal=[]

    for i in range(len(contours)):

        if hierarchy[0][i][0]==-1 and hierarchy[0][i][2]!=-1 and hierarchy[0][i][3] in total_list:

                if external_box_child[i]>20:

                    animal.append(contours[hierarchy[0][i][3]])
    return animal


def find_habitat(contours , hierarchy):

    '''
    Argument:

    contours  : A list consisting of the pixel value of each contour in an image.

    hierarchy : A numpy array of single element consiting a numpy array og length of contours each having 4 sub index,

    having value of Next , Previous , Child , Parent contours.


    Return:

    habitat : A similar type data as in contours but only having the contours inside which there is a habitats


    Description:


    It finds the boxes inside big box having a habitat image.


    '''


    parent_list=[hierarchy[0][i][3] for i in range(len(contours))]

    set_of_parent_list=set(parent_list)

    count_of_parent_list={parent_list.count(i):i for i in set_of_parent_list if hierarchy[0][i][3]==0}

    matrix_box=count_of_parent_list[max(count_of_parent_list.keys())]

    matrix_box,max(count_of_parent_list.keys())


    habitat=[]
    h=[]

    for cnt in range(len(contours)):

        if hierarchy[0][cnt][3] == matrix_box:

            h.append(cnt)

    for cnt in range(len(contours)):

        if hierarchy[0][cnt][3] in h and cv2.contourArea(contours[cnt])>400:

            habitat.append(contours[hierarchy[0][cnt][3]])

    return habitat , matrix_box

def generate_animal( image , animal, contours , hierarchy):

    '''
        Argument:

        image : input image

        animal : list of contours containing animals

        contours  : A list consisting of the pixel value of each contour in an image.

        hierarchy : A numpy array of single element consiting a numpy array og length of contours each having 4 sub index,

        having value of Next , Previous , Child , Parent contours.


        Return:

        None


        Description:


        It saves the each contour genrated as its position name as given in the rule book

    '''

    label_animal={
    0:'F1',
    1:'E1',
    2:'D1',
    3:'C1',
    4:'B1',
    5:'A1',
    6:'F2',
    7:'A2',
    8:'F3',
    9:'A3',
    10:'F4',
    11:'A4',
    12:'F5',
    13:'A5',
    14:'F6',
    15:'E6',
    16:'D6',
    17:'C6',
    18:'B6',
    19:'A6'
    }

    l=[]

    for i in range (len(contours)):

        if hierarchy[0][i][3]==0:

            l.append(contours[i])




    animal_list=[]

    for i in range(len(l)):

        if cv2.contourArea(l[i])>cv2.contourArea(contours[0])/3:
            continue

        animal_list.append(l[i])

    os.mkdir('Animals')
    os.chdir('Animals')

    for i in range(len(animal)):

        x1,y1,w1,h1=cv2.boundingRect(animal[i])

        a=int(x1+w1/2)
        b=int(y1+h1/2)

        for j in range(len(animal_list)):

            x,y,w,h=cv2.boundingRect(animal_list[j])

            if x < a < x+w and y < b < y+w:

                cv2.putText(image, label_animal[j] , (x+int(w*0.4)-4,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3 ,255)

                roi=image[y+10:y+h-10,x+10:x+w-10]

                cv2.imwrite(str(label_animal[j])+'.jpg',roi)

    os.chdir('..')


def generate_habitat(image , habitat , box):


    '''
            Argument:

            image : input image

            habitat : list of contours containing habitat

            box : contour value of the outer cotour containing the inner boxes


            Return:

            None


            Description:


            It saves the each contour genrated as its position name as given in the rule book

    '''


    os.mkdir('Habitats')
    os.chdir('Habitats')

    x1,y1,w,h=cv2.boundingRect(box)

    Points=[]

    c=int((h+w)/10)

    y=y1+h
    for i in range(5):
        x=x1
        for j in range(5):
            Points.append([x,x+c,y-c,y,i*5+j+1])
            x=x+c
        y=y-c


    for i in range(len(habitat)):
        x,y,w,h=cv2.boundingRect(habitat[i])
        for j in range(len(Points)):
            if Points[j][0] < x+w/2 < Points[j][1] and Points[j][2] < y+h/2 < Points[j][3]:
                cv2.putText(image, str(Points[j][4]), (x+int(w*0.4),y+int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,255,255))
                roi=image[y+15:y+h-15,x+15:x+w-15]
                cv2.imwrite(str(Points[j][4])+'.jpg',roi)

    os.chdir('..')

def save_contour(image , animal , habitat , contours , hierarchy , box):

    '''
        Argument:

        image : input image

        animal : list of contours containing animals

        contours  : A list consisting of the pixel value of each contour in an image.

        hierarchy : A numpy array of single element consiting a numpy array og length of contours each having 4 sub index,

        having value of Next , Previous , Child , Parent contours.

        habitat : list of contours containing habitat

        box : contour value of the outer cotour containing the inner boxes


        Return:

        None



    '''

    if os.path.exists('Images'):
        shutil.rmtree('Images')
    os.mkdir('Images')
    os.chdir('Images')

    generate_animal(image , animal , contours , hierarchy)
    generate_habitat(image , habitat , box )

    os.chdir('..')



def process_image(image,save):

    '''
    Argument:

    image : input image name
    save : LOcation of image to be saved



    Return:

    None


    Description : It process the input imahe and save the animals and habitat image respectively as their position name
    '''
    read_input_image = read_image(image)

    image, contours, hierarchy = find_contours(read_input_image)

    animal_contour = find_animal(contours , hierarchy)

    habitat_contour , matrix_box = find_habitat(contours , hierarchy)

    total = animal_contour + habitat_contour

    img = cv2.drawContours(read_input_image, total , -1 , (0,255,0), 3)

    save_contour(read_input_image,animal_contour , habitat_contour , contours , hierarchy , contours[matrix_box])

    if save:
        cv2.imwrite(save,read_input_image)
