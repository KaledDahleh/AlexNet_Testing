import torch
import torchvision.models as models
from PIL import Image
import os # interact with the os, in our case, specifically files
import torch.nn as nn
import json

# Path to your file
classIndexPath = "/Users/kaleddahleh/Desktop/workspace/repos/AlexNet_Testing/imagenet_class_index.json"

# Open and read the file
with open(classIndexPath, 'r') as file:
    dictionaryOfClasses = json.load(file)

# convert labels file to a list
openFile = open("classes.txt")
listOfClassNames = openFile.readlines()
for i in range(len(listOfClassNames)):
    listOfClassNames[i] = listOfClassNames[i].strip("\n")

# bring in alexnet model with pretrained weights and biasesl; we're not looking to train the model
model1 = models.alexnet(pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model1.to(device)

# this gets all the image prep we need
imageTransformations = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

imageFolderPath = "/Users/kaleddahleh/Downloads/ImageNet_Validation_Set"

openp = open(classIndexPath)

rd = openp.readlines()

correct = 0
incorrect = 0
total = 50000

with torch.inference_mode():
    listOfImageNames = os.listdir(imageFolderPath) # get the list of all image names
    listOfImageNames.sort()
    for imageName in listOfImageNames:
        imagePath = f'{imageFolderPath}/{imageName}' # combine the folder path and image name to get the image path
        # prepare the image for testing, these are AlexNet's documentation rules
        image = Image.open(imagePath) # open the image
        
        if image.mode == 'L':  # convert black and white to rgb
            image = image.convert("RGB")

        image = imageTransformations(image) # transform image to fit AlexNet testing documentation
        image = image.unsqueeze(0) # add an additional layer to the image tensor, batch size, because this is what ImageNet expects [N, C, H, W]
        image = image.to(device)

        prediction = model1(image) # forward pass

        probabilities = nn.functional.softmax(prediction[0], dim = 0) # softmax is a math operation that converts raw predictions to probabilities

        topProb, correspondingClass = torch.max(probabilities, dim = 0) # retrieve the highest probability score and its class

        predictedLabel = listOfClassNames[correspondingClass.item()]

        print(f'-----\n{predictedLabel}\nprobability: {topProb*100}%\nfilename: {imageName}')

        correctLabel = dictionaryOfClasses[str(correspondingClass.item())][1]

        if correctLabel == predictedLabel:
            print("CORRECT")
            correct+=1
        else:
            print("NOT CORRECT")
            incorrect+=1

        print(f'{correct} correct, {incorrect} incorrect')

    print(correct)
    print(incorrect)
    print(total)
    accuracy = f'{100*(correct/total)}%'
