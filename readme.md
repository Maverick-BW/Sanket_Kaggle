# Machine Learning Model Training and Evaluation-

This repository demonstrates techniques for training and evaluating machine learning models using Word2Vec feature extraction, logistic regression, and random forest classifiers. Key functionalities include:

Document Vectorization: 
Implements Word2Vec to generate document-level feature vectors.
Logistic Regression: 
Trains and evaluates a logistic regression model with accuracy and classification report metrics.
Random Forest Classifier: 
Trains and evaluates a random forest classifier for comparison.
Feature Matrix: Applies feature extraction on tokenized texts and constructs a feature matrix for testing

#Generating and Exporting Predictions

This section of the pipeline focuses on creating predictions and saving them for further use or submission:

Prediction DataFrame:
Generates a DataFrame  containing the model's predictions.


Submission File:
Combines predictions with associated IDs from the test dataset to create a submission-ready DataFrame (submission).
Saves the resulting file as submission.csv for evaluation or upload.

submission csv file creating and dowloading:
Includes functionality to interactively visualize the submission DataFrame using sheets.submission sheet download the predictions