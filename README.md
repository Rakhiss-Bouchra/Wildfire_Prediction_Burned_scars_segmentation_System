# Wildfire Prediction & Burned Scars Segmentation System

This project presents a system for the prediction and segmentation of burn scars using Python libraries and deep learning techniques to analyze satellite images.

## Datasets

In this project, we utilized two different datasets for prediction and burn scar segmentation.
- For prediction, you can access the dataset through the following link: https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset.


The original wildfire dataset used for prediction is sourced from the Government of Canada's Open Data Portal. For more details on the data collection process, you can find the notebook in this repository: https://github.com/Rakhiss-Bouchra/Wildfire_Data_Analysis

- For burned scars segmentation:
The dataset used in this project comprises satellite images paired with corresponding burn scar masks.

## Models

- Wildfire Prediction Model: This model is a Convolutional Neural Network (CNN) that has been trained on satellite imagery to determine whether a specific location is susceptible to a wildfire or not.

- Burned Scars Segmentation Model: This model is built upon the SegNet architecture, is trained with annotated satellite imagery to generate a mask that highlights the areas of burn scars within the image.

## Application 

The Streamlit application employs these two distinct deep learning models to create an integrated system for both wildfire risk prediction and burn scar segmentation. 
Users have the option to input the latitude and longitude coordinates of their area of interest, after which the application will provide a probability of the area being at risk for a wildfire. 
Additionally, for burn scar segmentation, users can upload a satellite image to visualize the delineation of burn scars.

<img width="681" alt="Interface" src="https://github.com/Rakhiss-Bouchra/Wildfire-Prediction-burn-scars-segmentation-system/assets/100072520/122d4a40-9571-43ee-a9db-b40fa68fd860">
