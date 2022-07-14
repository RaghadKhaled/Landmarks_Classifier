# Detect and Classify Landmarks in images using CNN

![Screenshot (631)](https://user-images.githubusercontent.com/68460588/178982767-7203da3b-bff9-49e0-a676-437507a924d7.png)


--- 

## Overview

Photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.
If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.
In this project, I work to addressing this problem by building CNN models to automatically predict the most likely locations where the image was taken based on any landmarks depicted in the image.
I did this project as a part of Deep Learning Nanodegree from Udacity. 

--- 

## Dataset

The landmark images are a subset of the Google Landmarks Dataset v2. It has different 50 landmarks from all of the world. You can download the datatset by [this link](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip). You can find more information for the full dataset [on Kaggle](https://www.kaggle.com/google/google-landmarks-dataset).

--- 


## Project Steps

The high-level steps of the project include:
1.	Build a CNN to Classify Landmarks (from Scratch) 
2.	Build a CNN to Classify Landmarks (using Transfer Learning)  – I used VGG16 since my dataset has similar feature of the dataset that trained on the VGG16. Therefore, most or all of the pre-trained neural network layers already contain relevant information about the my dataset. I Use all but the last fully connected layer as a fixed feature extractor and I define a new, final classification layer to match the number of classes of my task. I randomize the weights of the new fully connected layer and freeze all the weights from the pre-trained network then I train the network to update the weights of the new fully connected layer.
3.	Create Landmark Prediction Algorithm – this algorithm take the best model I created to make interface for any user-supplied image as input and suggest the top k most relevant landmarks from 50 possible landmarks from across the world. 
All the implementation details are in jupyter notebook file ``` landmark.ipynb```.

--- 

## Results

I test the model from image in my computer, the results are showed below. 

![Untitled - Frame 1](https://user-images.githubusercontent.com/68460588/178982989-dc64563e-18b8-4b21-accf-13cce068fb7d.jpg)


---

## Requirements	
- Python 3.7
- Numpy 
- Torch
- Torchvision
- PIL
- OpenCV
- Matplotlib 
- Jupyter Notebook
- Maybe needs to use GPU


---

## Running the project
The whole project is located in the jupyter notebook file ``` landmark.ipynb ```, you can use the Anaconda environment to open the Jupyter Notebook and install the requirement.
