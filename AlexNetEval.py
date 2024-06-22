import torch
import torchvision.models as models
from PIL import Image
import os # interact with the os, in our case, specifically files

# bring in alexnet model with pretrained weights and biasesl; we're not looking to train the model
model1 = models.alexnet(pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model1.to(device)

# this gets all the image prep we need
imageTransformations = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

imageFolderPath = "/Users/kaleddahleh/Downloads/test"

with torch.inference_mode():
    listOfImageNames = os.listdir(imageFolderPath) # get the list of all image names
    for imageName in listOfImageNames:
        imagePath = f'{imageFolderPath}/{imageName}' # combine the folder path and image name to get the image path
        # prepare the image for testing, these are AlexNet's documentation rules
        image = Image.open(imagePath) # open the image
        image = imageTransformations(image) # transform image to fit AlexNet testing documentation
        image = image.unsqueeze(0) # add an additional layer to the image tensor, batch size, because this is what ImageNet expects [N, C, H, W]
        image = image.to(device)

        prediction = model1(image) # forward pass

        print(prediction) # data is still raw

    
