import torch
import torchvision.models as models
from PIL import Image
import os
import torch.nn as nn
import json
# ---------------------------------------------------------------------------------------------------------------------------------------
# -------------------- convert json map of classes into a python dictionary ----------------------------------------
classIndexPath = "/Users/kaleddahleh/Desktop/workspace/repos/AlexNet_Testing/ImageNet_Class_Map.json" # this is the map of the 1000 class indices and their corresponding ID and label
with open(classIndexPath, 'r') as file:
    dictionaryOfClasses = json.load(file)
codesAndLabels = list(dictionaryOfClasses.values())
newdict = {}
for codeAndLabel in codesAndLabels:
    newdict[codeAndLabel[0]]=codeAndLabel[1]
# ---------------------------------------------------------------------------------------------------------------------------------------
# -------------------- bring in alexnet model with pretrained weights and biases since we're not looking to train the model -------------
model1 = models.alexnet(pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model1.to(device)
# ---------------------------------------------------------------------------------------------------------------------------------------
# -------------------- this is a special pytorch class that adjusts images so they fit AlexNet criteria --------------------
imageTransformations = models.AlexNet_Weights.IMAGENET1K_V1.transforms()
# ---------------------------------------------------------------------------------------------------------------------------------------
# -------------------- Load list of 50k AlexNet validation images from file path -----------------
imageFolderPath = "/Users/kaleddahleh/Downloads/ImageNet_Validation_Set"
listOfImageNames = os.listdir(imageFolderPath) # get the list of all image names
listOfImageNames.sort()
# ---------------------------------------------------------------------------------------------------------------------------------------
# -------- Prepare bounding box annotations to identify each image's correct ID ---------------------------------------------------------------------------------------
Bounding_Box_Annotation_Folder_Path = "/Users/kaleddahleh/Downloads/val"
Bounding_Box_Annotation_Folder = os.listdir(Bounding_Box_Annotation_Folder_Path)
Bounding_Box_Annotation_Folder.sort()
# ---------------------------------------------------------------------------------------------------------------------------------------
with torch.inference_mode():
    # -------------------- initialize values -----------------
    successfullyPredictedImages = 0
    unsuccessfullyPredictedImages = 0
    imageNum = 0 # index for the image we are currently testing
    # -------------------------------------------------------------
    for imageName in listOfImageNames:
        imagePath = f'{imageFolderPath}/{imageName}'
        # -------- prepare the image for testing, these are AlexNet's documentation rules -------------
        image = Image.open(imagePath)
        if image.mode != 'RGB': 
            image = image.convert('RGB')
        image = imageTransformations(image)
        image = image.unsqueeze(0) # add an additional layer to the image tensor, batch size, because this is what ImageNet expects [N, C, H, W]
        image = image.to(device)
        # ---------------------------------------------------------------------------------------------
        # ---------- Forward pass to retrieve predicted label -----------------------------
        prediction = model1(image)
        probabilities = nn.functional.softmax(prediction[0], dim = 0) # softmax is a math operation that converts raw predictions to probabilities
        topProb, correspondingClass = torch.max(probabilities, dim = 0) # retrieve the highest probability score and its class(number not label)
        predictedLabel = dictionaryOfClasses[str(correspondingClass.item())][1] # go from class number to label
        # ---------------------------------------------------------------------------------------------
        # ---------- Retrieve the correct image label ---------------------------------------
        Validation_Images_Folder_Path = f'{Bounding_Box_Annotation_Folder_Path}/{Bounding_Box_Annotation_Folder[imageNum]}'
        openFile = open(Validation_Images_Folder_Path)
        listOfImageAttributes = openFile.readlines()
        imageID = listOfImageAttributes[13][8:17]
        correctLabel = newdict[imageID]
        # ---------------------------------------------------------------------------------------------
        # ------ RESULTS ------------------------------------------------------------
        print(f'correct label: {newdict[imageID]}\npredicted label: {predictedLabel}\nImage Number: {imageNum}')
        if correctLabel == predictedLabel:
            print("prediction was: CORRECT")
            successfullyPredictedImages+=1
        else:
            print("prediction was: WRONG")
            unsuccessfullyPredictedImages +=1 
        print("----------------------------------------")
        imageNum += 1
    print(f'{successfullyPredictedImages} correct\n{unsuccessfullyPredictedImages} incorrect\nAccuracy: {100*successfullyPredictedImages/(successfullyPredictedImages+unsuccessfullyPredictedImages)}%')
    
