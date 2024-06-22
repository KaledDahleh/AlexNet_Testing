# Inference/Test of AlexNet model on the ImageNet dataset

## need to write code to run inferences, as well as download the weights and biases pointing to the data

### DEPENDENCIES

    - ImageNet test images (13gb) can be downloaded from [this](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php#images) link, once logged into [image-net.org](image-net.org)

    - pip install torch torchvision pillow
        - you need PyTorch 1.9 or later to use "with torch.inference_mode()", otherwise, modify the code to "with torch.no_grad()"

    - ImageNet dataset --> found on ImageNet website, requires login to access dataset
