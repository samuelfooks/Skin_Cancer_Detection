## Skin_Cancer_Detection
Final Machine Learning Project @ BeCode February 2023

For this project, the objective was to create an application that can evaluate the risk that a submitted image of a mole/observable skin growth
is a cancerous or not. Doing this by making use of imaging processing and machine learning techniques used in Computer Vision.

## Description

This project used the HAM_10000 dataset from kaggle
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

My take on this project was to take advantage of already existent examples, and refine those techniques used.  Mainly by making use of Transfer Learning from already trained image processing neural networks('Xception', 'ResNet', etc)
I was able to produce a model that has good predictive power, and rapid deployability.
(False negatives 67/376(17.7%))

The general structure of the project is as follows:

    1.EDA_metadata.py is a file that performs some basic Exploratory Data Analysis, mainly highlighting the imbalanced nature of the dataset and the lower incidences of cancerous skin. Here there is the option to explore the dataset
    and change the dataset to be used in the modelling.

    2.data_process_split.py uses the refined dataset from EDA_cleaning to prepare the training, test and validation data and images that will be used to train and evaluate the neural network

    3.model.py uses the the split and refined dataset and trains a neural network based, using a given "base_model" from tensorflow, with also the option of using no pre trained base model. It also saves the trained model as well as the model weights after each epoch. It then evaluates the performance of that model
    based on the train, test, validation dataset from data_process_split.py

    4.App.py is the streamlit app file that runs the photo based prediction application.  The app can be deployed locally or online.  The app loads a pretrained keras model, asks for a photo(uploaded or directly taken), makes a prediction if the photo contains a cancerous skin growth or not, and gives the probability of that outcome.  

Further work will focus on increasing the accuracy of the pretrained models by improving model.py and developing a geographic link in the App.py to give the locations of nearest
physicians if the resulting prediction is cancerous.
## Getting Started
First download the dataset from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

After doing some Exploratory Data Analyis, create a metadata csv.  Run the model.py file, choosing the appropriate parameters(oversampling, base_model, etc)  Once you have a trained model, check through the evaluation metrics and also check to see that it can make predictions from the saved test csv files.

Then deploy the model in App.py, locally or online.

Enjoy!

### Dependencies

requirements.txt
run in a conda environment to keep track of dependencies
### Installing

run in python 3.7
pip intall -r requirements.txt
### Executing program

Downloading the dataset from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
Run the data processing files. Then run the model file to build, save, and evaluate your own neural network model

Use this model then in the App.py and deploy on any platform of choice

### Authors

Samuel Fooks
### Version History

* 0.1
    * Initial Release
## License

Free license, open source
## Acknowledgments
Thanks BeCode Coaches!
Louis D.V.
Chrysanthi K.

Great starting basis and examples for models to apply transfer learning from https://github.com/faniabdullah/bangkit-final-project