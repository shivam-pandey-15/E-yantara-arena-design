
'''

* Team_ID    :   HC#3344

* Filename   :   main.py

* Theme      :   E-yrc Homecoming



* Functions Used :

 parsing()



'''

import argparse

from Image_formation.generate_image import process_image

from model import evaluate

import shutil

def parsing():


          '''


          Arguments: None

          Returns:
                     args.imgname,args.s,args.c:

                      A tuple of 3 values consisting Input image name , Output image_name , Location of saving contours



          Description:

                The function takes command Line argument and performs the following actions as they are mentioned in task details
          '''



          parser = argparse.ArgumentParser(add_help=False)

          #Image
          parser.add_argument("img",help="Input Animal Image",type=str)
          #Save Image
          parser.add_argument("-s",help="Input Habitat Image",type=str)
          #Animal model
          parser.add_argument("--amod",help="Load Animal model",type=str)
          #Habitat model
          parser.add_argument("--hmod",help="Load Habitat model",type=str)

          args = parser.parse_args()


          return args.img , args.s , args.amod , args.hmod

if __name__ == "__main__":
    image , save , animal_model , habitat_model = parsing()

    process_image(image,save)

    if animal_model==None:
        animal_model = 'best_animal_resnet18.pth'


    if habitat_model == None :

        habitat_model = 'best_habitat_resnet18.pth'

    animal,habitat = evaluate(animal_model , habitat_model)

    print('Animals','\n\n')
    print(animal)
    print('\n\nHabitat','\n\n')
    print(habitat)

    shutil.rmtree('Images')

    final = {}
    final.update(animal)
    final.update(habitat)
    print(final)
