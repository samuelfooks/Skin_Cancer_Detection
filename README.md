ReadMe

For this project, the objective was to create an application that can evaluate the risk that a submitted image of a mole/observable skin growth
is a cancerous or not. Doing this by making use of imaging processing and machine learning techniques used in Computer Vision.


This project used the HAM_10000 dataset from kaggle
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

My take on this project was to take advantage of already existent examples, from similar competitions or this one itself, and refine those
techniques used.  Mainly by making use of Transfer Learning from already trained image processing neural networks('Xception', 'ResNet', etc)
I was able to produce a model that has great predictive power, and rapid deployability.

The general structure of the project is as follows:

    EDA_cleaning.py is a file that performs some basic Exploratory Data Analysis, mainly highlighting
    the imbalanced nature of the dataset and the lower incidences of cancerous skin

    data_process_split.py uses the refined dataset from EDA_cleaning to prepare the training, test and validation data and
    images that will be used to train and evaluate the neural network

    model.py uses the the split and refined dataset and trains a neural network based, using a given "base_model" from
    tensorflow, with also the option of using no pre trained base model. It then evaluates the performance of that model
    based on the train, test, validation dataset from data_process_split.py

    App.py is the streamlit file that runs the application



