# AlexNet Image Classification

**IMPORTANT --> Ensure you update the paths in the script to reflect your directory structure accurately! More on this in the "Setup" heading**

This project uses the pretrained AlexNet model to classify images from the ImageNet val set (50k images). The script loads images, processes them, runs them through the model, and compares the predictions to the correct labels.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Pillow

## Setup

1. **Clone the repository**

   - git clone https://github.com/yourusername/AlexNet_Testing.git
   - cd AlexNet_Testing

2. **Install requirements**

   - pip install -r requirements.txt

3. **Download ImageNet Class Index**

   - Download the "ImageNet_Class_Map.json" file and place it in the root directory of this project.

4. **Prepare ImageNet Validation Set**

   - Ensure that your ImageNet validation set images are placed in a directory and update the "imageFolderPath" variable in the script with this path.

5. **Prepare Bounding Box Annotations**

   - Ensure that your correct labels (in XML format) are placed in a directory and update the "Bounding_Box_Annotation_Folder_Path" variable in the script with this path.

## Running the Script

    python3 AlexNetEval.py
