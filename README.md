# Nutrition-Detection-System-using-Deep-Learning-
<div style="border-radius:12px; padding: 20px; background-color: #d5d5ed; font-size:120%; text-align:center">


## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Workflow](#workflow)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Used](#model-used)
6. [Training](#training)
7. [Results](#results)  
8. [Future Work](#future-work)
9. [Conclusion](#conclusion)
# Introduction
 Modern world demands for concrete methods which help study and analyse the nutritional value and content in
 food, with a peak in dietary-related health issues as well. Harnessing Artificial Intelligence along with deep
 learning methodologies like convolutional neural networks to help in food recognition and additionally detect
 nutritional value for the same. Paper proposes an approach to study the accurate detection of nutrients in food
 cuisines using traditional as well as precisely modified variants of convolutional neural network models.
 Through a particular food image input the proposed model uses methods like feature engineering, extraction to
 precisely recognise the cuisine it belongs to and furthermore estimates its nutritional composition such as calorie
 count, proteins, fat etc.

 Implementing Deep Learning Techniques for accurate nutrient detection in different food cuisines
 Modelcan adapt and handle diverse cuisines ranging from seafood, Chinese, Mughlai etc. Model’s
 versatility ensures that regardless of food plating presentation styles or camera angles of the food
 image, the model achieves great results in detecting nutrients in all such scenarios.
 FoodIndustry, Healthcare professionals and diet conscious individuals understand the time and efforts
 which go behind accurate & detailed nutrient analysis. Our model efficiently finds a solution to save
 their time and efforts significantly and improve nutrient analysis performance.

# Installation
To run the code in this project, you will need the following dependencies installed:

1.Python: Version 3.6 or higher
2.TensorFlow: Version 2.0 or higher
3.NumPy: For numerical computations
4.Matplotlib: For data visualization
5.Pandas: For data manipulation and preprocessing
# Workflow 
![Flowchart Building (1)](https://github.com/shambhavi1010/Nutrition-Detection-System-using-Deep-Learning-/blob/main/DL%20Flow.png)



# Dataset Preparation

Dataset collected by the author consists of over 600-700 images of food cuisines of over 11 classes. Every class
 represents a specific type of food item. Images of every class are diverse in nature, various camera angles,
 presentation style, colour gradient.  Every image downloaded from the extension has its own file path
 which has been collected to create a csv file format consisting of all file paths and its labels. 

  Necessary python libraries for preprocessing and data cleaning are imported. Since before the implementation of
 deep learning algorithms, models must work on splitting data into training and testing. Model deals with image
 dataset completely so it uses Encoding methods of One Hot Encoder and Label Encoder for the same.
 Author makes use of the NumPy library for random seed data generation to feed to the potential network. Every
 image size is hard coded to be of size 224 x 224, batch size 32 and the model architecture is designed to work
 for over 40 epochs.
 Since the dataset consists of over 12 classes of food cuisines, the model needs to loop through the sub classes to
 extract features for every food image under each class. Author implements several loops for the model to go
 through the data.
 Author uses Label Encoder and One Hot Encoding methods to encode the labels of every image into encodings.
 Firstly, labels are converted to numerical values.


# Model Used

Convolutional Neural Network ( Three varying modification)
ResNet
AlexNet
VGG Network
Mobile Net:
 This is a relatively lightweight deep learning architecture which is originally optimized for mobile
 devices. Model’s depth wise separable convolutions reduce the computational power and the model’s
 size. Mobile Net is known for its ability to balance accuracy and efficiency. Image Data Generator
 function executed during pre-processing generates images to feed to the network from the training set.
 Model must exclude the top classification layers. Mobile Net uses two dense layers for further
 classification. Dense layer has 128 units.
 Adam optimizer and sparse categorical cross entropy is used to carry out the network.






# Training

### Data Preparation
Before training can begin, the labels are converted to numerical values using one-hot encoding method
- **Data Splitting  & Augmentation**: DataFrame with file paths and labels, then encodes the labels using one-hot encoding. It splits the dataset into training, validation, and test sets. Data augmentation is applied to the training set, while validation and test sets are only rescaled. Image data generators are created for each set, preparing the images for model training with specified transformations and rescaling.

### Training Steps
The training process involves several steps that are executed in each epoch:

1. **Respective Deep Learning Model implementation on obtained data above**:
   - We train varying models, CNN, CNN with PSO, VGG16, RESNET, ALEXNET and finally MobileNet which gives the best accuracy amongst all.

2. **Evaluating Results**:
   - Classification Report for performance of every model and test accuracy evaluated. 

3 . **Mobile Net Features **:
   - The sequential model trained as the base model using global average pooling and relu activation function. Model runs for over 40 epochs on the generated data.

This detailed training process is deployed using simple streamlit and for the nutrient analysis segment  since the data is quite small-scale and consists of merely 12 classes of food cuisines and has a super
 limited data under each class. For the classification of food items to be as accurate as possible from the
 input image using deep learning models, our model learns features like colour, shape, borders, and
 other factors. For the simplicity of food nutrient detection part of the project. Fixed serve size is
 determined for every cuisine class in our data and corresponding to this serve size, nutrients are derived
 using a lookup table

# Results

Classification Report of MobileNet
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/7db704e6-f3bb-4be1-b4f8-861d6717bfa8)

Data visualized
![download (1) (2)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/6a99207d-e75e-4949-a0c7-38e782d8f778)




Confusion Matrix for MobileNet
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/37171f36-179e-41d2-b143-c8ba73ae4e36)






# Future Work
 The proposed model currently makes its predictions only on a limited dataset consisting of simple dishes like
 French fries, fish, rice etc. To create a more varied data of these classes the author has collected a huge chunk of
 images available as well as opted for ways like image augmentation, rotation etc. This makes the training data
 diverse, and the model is introduced to new cases at several iterations. Future scope for this model will be to
 make new nutrient analysis estimation on a more varied dataset, also targeting specific vegetable cuisines as
 food items like vegetables, rice, pulses take up most of our diet daily. 

# Conclusion

or a problem domain like ‘Nutrient detection of food cuisines’ using only an input image might be a
 complicated task for a simple neural network or classification model. Although every deep learning
 architecture uses a SoftMax in the end of its model for the final classification, but as we deal with
 images Convolutional neural network provides you with features of your food image and using several
 parameters such as filters, feature maps, pooling, stride and finally flattening the values obtained into a
 one dimensional array of numerical values which is known to be the fully connected layer is then fed to
 the SoftMax for classification. Without feature engineering and extraction carrying out food recognition
 is near to impossible. 

### Achievements
- **Successful detection of food cuisine**:MobileNet implements necessary parameter changes and detects the food in the iimage provided.
- **Nutrient Value Approximate Estimate**:Standard serve size basis for analysing the possible nutrient content.

