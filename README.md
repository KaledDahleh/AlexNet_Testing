# This is useful for anyone looking to understand how model testing is made at the surface level

    - This guide is useful for anyone looking to understand how model testing is conducted at a surface level. As someone exploring deep learning with PyTorch, I found this step important in my learning journey. I've tried to make this as simple as possible for my foundational understanding and plan to expand on it as I see beneficial.

## Validation of AlexNet model on the ImageNet dataset

### need to write code to run validation, as well as download the weights and biases pointing to the data

#### DEPENDENCIES

    - ImageNet validation images (6gb, 50k images) can be downloaded from [this](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php#images) link, then nagivate to "Validation images (all tasks)", once logged into [image-net.org](image-net.org)
        - download the corresponding image labels from the same link --> "Development kit (Task 1 & 2)"

    - pip install torch torchvision pillow
        - you need PyTorch 1.9 or later to use "with torch.inference_mode()", otherwise, modify the code to "with torch.no_grad()"

    - ImageNet dataset --> found on ImageNet website, requires login to access dataset
