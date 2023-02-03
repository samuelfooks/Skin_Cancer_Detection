
## Skin_Cancer_Detection

For this project, the objective was to create an application that can evaluate the risk that a submitted image of a mole/observable skin growth
is a cancerous or not. Doing this by making use of imaging processing and machine learning techniques used in Computer Vision.

## Description

This project used the HAM_10000 dataset from kaggle
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

My take on this project was to take advantage of already existent examples, from similar competitions or this one itself, and refine those
techniques used.  Mainly by making use of Transfer Learning from already trained image processing neural networks('Xception', 'ResNet', etc)
I was able to produce a model that has great predictive power, and rapid deployability.

The general structure of the project is as follows:

    1.EDA_metadata.py is a file that performs some basic Exploratory Data Analysis, mainly highlighting
    the imbalanced nature of the dataset and the lower incidences of cancerous skin. Here there is the option to explore the dataset
    and change the dataset to be used in the modelling.

    2.data_process_split.py uses the refined dataset from EDA_cleaning to prepare the training, test and validation data and
    images that will be used to train and evaluate the neural network

    3.model.py uses the the split and refined dataset and trains a neural network based, using a given "base_model" from
    tensorflow, with also the option of using no pre trained base model. It then evaluates the performance of that model
    based on the train, test, validation dataset from data_process_split.py

    4.App.py is the streamlit app file that runs the application

## Getting Started

If you would like to train, use and evaluate your own model that makes predictions based on uploaded photos you can using the first 3 files.
Or you can just deploy the App.py using streamlit
### Dependencies

requirements.txt

### Installing

run in python 3.7

### Executing program

Downloading the dataset from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

Then open the python files and go for it!!


### Authors

Samuel Fooks

### Version History

* 0.1
    * Initial Release

## License
This project is not licensed

## Acknowledgments

Great starting basis and examples for models to apply transfer learning from https://github.com/faniabdullah/bangkit-final-project
