# Dog-Vision 🐕
This repository contains the code for an end-to-end multi-class image classifier built using TensorFlow 2.0 and TensorFlow Hub. The goal of this project is to classify images of dogs into one of 120 breeds. The project is based on [Kaggle’s Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/overview) competition. The model achieved a ranking of 152 out of 1281 participants on the leaderboard.

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Data](#data)
3. [Evaluation](#evaluation)
4. [Features](#features)
5. [Preprocessing](#preprocessing)
6. [Modeling](#modeling)
7. [Prediction](#prediction)
8. [Results and Kaggle Submission](#results-and-kaggle-submission)
9. [TensorFlow WorkFlow](#tensorFlow-workflow)
 
## Problem Definition
The challenge is to identify the breed of a dog given its image. This is a supervised learning task in which the input is an image of a dog, and the output is the predicted dog breed. Given that there are 120 possible breeds, this is a multi-class classification problem.

## Data
The dataset used for this project comes from [Kaggle's Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/overview) competition, and it contains:
- **Training Set**: 10,222 labeled images with their corresponding dog breeds.
- **Test Set**: 10,357 unlabeled images for which the goal is to predict the breed.
- **Breed Classes**: 120 unique dog breeds, forming the target classes for the classification task.

## Evaluation
- The project is evaluated using **Multi-Class Log Loss**. This metric compares the predicted probabilities for each dog breed against the true breed labels. 
- The goal is to minimize log loss, where a lower value indicates that the model is predicting probabilities closer to the true class.

## Features
The data for this project comes from the Kaggle Dog Breed Identification Competition. It includes labeled images of dogs that belong to 120 different breeds.
- **Input**:
  - **Images**: Each input is a dog image, which is resized to a consistent size of 224x224 pixels for model training and inference.
- **Output**:
  - **Predicted Breed Probabilities**: The model outputs a probability distribution across 120 possible dog breeds for each input image. 

## Preprocessing
Key preprocessing steps include:
- **Loading Images**: The training and test images are loaded into memory, and paths are stored in a list.
- **Image resizing**: All images are resized to 224x224 pixels for consistency.
- **Normalization**: Pixel values are scaled between 0 and 1 to improve model training.
- **Tensor Conversion**: Images are converted to tensors, the numerical format required by TensorFlow models.
- **Creating a Validation Set**: Since Kaggle’s dataset does not provide a validation set, I created one by splitting 20% of the training set for validation. This split allows the model’s performance to be monitored during training.

## Modeling
- **Model Architecture**
For the image classification model, I used transfer learning with MobileNetV2, a lightweight and highly efficient pre-trained deep learning model available through TensorFlow Hub. The steps in the modeling process include:
  - **Input Layer**: Accepts 224x224x3 image tensors.
  - **Base Model**: A pre-trained MobileNetV2 model, used for feature extraction. This model was pre-trained on the ImageNet dataset.
  - **Dense Output Layer**: Contains 120 units (one for each dog breed), with a softmax activation to output the predicted breed probabilities.
- **Compilation:**
The model is compiled using:
- **Optimizer**: Adam with a learning rate of 0.0001.
- **Loss Function**: Categorical cross-entropy, appropriate for multi-class classification.
- **Metrics**: Accuracy, used to monitor model performance during training.
**Training**:
- **Batch Size**: The model was trained with a batch size of 32.
- **Epochs**: Training was run for 20-30 epochs, though early stopping was employed to halt training if validation loss stopped improving.
**Callbacks:**
- **TensorBoard**: To track the progress of our model.
- **EarlyStopping**: Stops training when validation loss does not improve for 5 consecutive epochs to prevent overfitting.

## Prediction
Once the model was optimized, it was used to make predictions on the unlabeled test set provided by Kaggle. The steps for making predictions include:
- **Loading and Preprocessing Test Images**: Test images were preprocessed using the same steps as the training images (resizing, normalization, tensor conversion).
- **Generating Predictions**: The trained model was used to generate a probability distribution over the 120 dog breeds for each test image.

**Custom Image Predictions**:
I also developed functionality to predict dog breeds from custom images:
- Upload and preprocess custom image.
- Predict breed: The model outputs the top predicted breed along with the associated probabilities.

## Results and Kaggle Submission
The final step in the project was preparing the submission for the Kaggle competition:
- **Create a DataFrame**: Consists of the image IDs from the test set and the predicted probabilities for each of the 120 dog breeds.
- **Export as CSV**: The DataFrame was saved as submission.csv, in the format required by Kaggle.
- **Submission**: The CSV file was uploaded to Kaggle for evaluation.
My final ranking on the Kaggle leaderboard was 152nd out of 1281 participants, demonstrating the effectiveness of the transfer learning approach.

## TensorFlow Workflow
1. Get data ready (turn into Tensors)
2. Pick a model (to suit your problem)
3. Fit the model to the data and make a prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload your trained model
