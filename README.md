# Planet Disease Classifier

# Planet Disease Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Transfer Learning with MobileNet](#transfer-learning-with-mobilenet)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Project Overview

The **Planet Disease Classifier** is a deep learning project aimed at predicting diseases in plants by analyzing images of plant leaves. This project utilizes a convolutional neural network (CNN) architecture, specifically MobileNet, combined with transfer learning techniques to classify 38 different diseases across 14 crop types. The goal is to provide an efficient and accurate tool for farmers and researchers to diagnose plant diseases early, potentially saving crops and improving yield.

## Dataset

The dataset used in this project consists of images of plant leaves categorized into 38 different disease classes. The dataset encompasses 14 different crops, ensuring a wide variety of disease and plant types are represented.

### Dataset Details
- **Number of Classes**: 38 diseases
- **Number of Crops**: 14 different crops
- **Data Source**: [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset]
- **Sample Images**:
      ![image](https://github.com/user-attachments/assets/9070cce5-267a-4cd1-933a-8fb12e04fd8d)


### Data Preprocessing
- Image resizing to `256x256` pixels.
- Normalization of pixel values.
- Data augmentation techniques (e.g., rotation, flipping, zooming) to increase dataset variability and improve model generalization.

## Model Architecture

The core of this project is the **MobileNet** CNN architecture, a lightweight model designed for efficient computation on mobile devices. Given the constraints of deployment in agricultural settings, MobileNet provides a balance between accuracy and computational efficiency.

### MobileNet Features
- Depthwise separable convolutions to reduce the number of parameters.
- Global average pooling layer before the fully connected layers to minimize overfitting.
- Pretrained on the ImageNet dataset, enabling transfer learning.

## Transfer Learning with MobileNet

Transfer learning was employed by fine-tuning the MobileNet model, pretrained on the ImageNet dataset. The pretrained weights allow the model to utilize the feature extraction capabilities learned from millions of images, while the final layers were retrained on our specific dataset.

### Transfer Learning Process
- **Frozen Layers**: Initial layers were frozen to retain the pre-learned features.
- **Custom Layers**: A new fully connected layer was added to classify the 38 disease classes.
- **Optimization**: The model was compiled with the Adam optimizer and a learning rate of `1e-4`.

## Training and Evaluation

The model was trained on the processed dataset with the following parameters:

- **Batch Size**: 64
- **Epochs**: 12
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Recall

### Validation
- **Validation Accuracy**: 0.9356
- **Validation Recall**: 0.9286

## Results

The Planet Disease Classifier achieved the following results:

- **Training Accuracy**: 94.02%
- **Training Recall**: 93.17%
- **Validation Accuracy**: 93.56%
- **Validation Recall**: 92.86%

- ![image](https://github.com/user-attachments/assets/675a13f8-9820-461f-a6bb-2d51819216bb)
- ![image](https://github.com/user-attachments/assets/47412a8a-f0d4-4ac1-92ee-02ba262d040a)



The model demonstrates strong performance in identifying a wide range of plant diseases, making it a valuable tool for early detection and prevention.


## Acknowledgements

- **Dataset**: [[Dataset Source](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)]
- **Pretrained Model**: MobileNet pretrained on ImageNet.
- **Tools & Libraries**: TensorFlow, NumPy, OpenCV.
