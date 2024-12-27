## Model Description and Results

This project demonstrates the application of machine learning models to a classification problem using logistic regression and random forest classifiers. Below are the key steps and results:
### Document Vectorization: 
Implements Word2Vec to generate document-level feature vectors.

### 1. **Data Splitting**
The dataset is split into training and testing sets using `train_test_split()` with a test size of 20%. The target variable `y` and the features `X` are used for the split.

### 2. **Logistic Regression**
- Trains and evaluates a logistic regression model with accuracy and classification report metrics.
- A logistic regression model is trained using the training set (`X_train` and `y_train`).
- The model's predictions on the test set (`X_test`) are evaluated using `accuracy_score` and `classification_report`.
- **Accuracy**: 58.96%
- **Classification Report**: 
  - Precision (class 0): 0.59
  - Recall (class 0): 0.98
  - Precision (class 1): 0.68
  - Recall (class 1): 0.07

### 3. **Random Forest Classifier**
- Trains and evaluates a random forest classifier for comparison.
- A random forest classifier is trained on the same training set.
- The predictions on the test set are evaluated with the same metrics.
- **Accuracy**: 70.72%
- **Classification Report**: 
  - Precision (class 0): 0.70
  - Recall (class 0): 0.87
  - Precision (class 1): 0.73
  - Recall (class 1): 0.49

### 4. **Prediction on Test Data**
- The trained logistic regression model is used to predict the target variable on a new dataset (`X_test_x`).
- The predictions are stored in a `DataFrame` and saved to a CSV file.





# Machine Learning Evaluation -

This repository demonstrates techniques for training and evaluating machine learning models using Word2Vec feature extraction, logistic regression, and random forest classifiers. Key functionalities include:

#Generating and Exporting Predictions

This section of the pipeline focuses on creating predictions and saving them for further use or submission:

Prediction DataFrame:
Generates a DataFrame  containing the model's predictions.

Submission File:
Combines predictions with associated IDs from the test dataset to create a submission-ready DataFrame (submission).
Saves the resulting file as submission.csv for evaluation or upload.

Submission csv file creating and dowloading:
Includes functionality to interactively visualize the submission DataFrame using sheets.submission sheet download the predictions


