# Train data (cleaning & Processing):

(Prajwal Kolhe & Shreyash Vibhute)


| **Step** | **Process Description**                                                                                      | **Output/Visualization**                                                                   |
|----------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 1        | Import necessary libraries such as Pandas, Numpy, Matplotlib, Seaborn, NLTK, and Scikit-learn.               | Libraries successfully imported.                                                         |
| 2        | Load the dataset `Disaster_train.csv` into a DataFrame.                                                      | Dataset loaded into memory as `data_train`.                                               |
| 3        | Display dataset structure using `info()` and preview using `head()`.                                         | Text output showing column names, data types, and first few rows.                         |
| 4        | Generate summary statistics using `describe()`.                                                              | Table showing count, mean, std, min, quartiles, and max for numeric columns.              |
| 5        | Identify and count missing values using `isnull().sum()`.                                                    | Table displaying the count of missing values for all columns.                             |
| 6        | Fill missing values in `keyword` and `location` columns with 'none' and 'unknown', respectively.             | Missing values replaced, confirmed by rechecking with `isnull().sum()`.                   |
| 7        | Define a `clean_text` function to preprocess text (remove URLs, hashtags, special characters, etc.).         | Function applied, resulting in cleaned text stored in a new column `cleaned_text`.        |
| 8        | Check for remaining missing values in `keyword` and `location` columns.                                      | Verified that missing values are handled.                                                |
| 9        | Visualize the label distribution using `sns.countplot()`.                                                    | Bar chart showing the distribution of target labels (e.g., 0 vs. 1).                      |
| 10       | Compute text lengths and add as a new column `text_length`.                                                  | New column `text_length` created.                                                        |
| 11       | Plot text length distribution using `sns.histplot()`.                                                        | Histogram showing the distribution of text lengths.                                       |
| 12       | Tokenize text and calculate word frequencies.                                                                | List of most common words created, stored as `common_words`.                              |
| 13       | Remove stopwords from `cleaned_text` using a custom function.                                                | Sample rows showing text before and after stopwords removal.                              |
| 14       | Apply lemmatization to further clean the text.                                                               | Sample rows showing text before and after lemmatization.                                  |
| 15       | Tokenize cleaned text into a list of words for Word2Vec input.                                               | Tokenized text prepared for Word2Vec training.                                            |
| 16       | Train a Word2Vec model on the tokenized text.                                                                | Model trained, and vocabulary size displayed.                                             |
| 17       | Extract Word2Vec features and prepare the dataset (`X` and `y`) for further modeling.                        | `X` (features) and `y` (labels) prepared, dimensions confirmed.                          |

# Test Data (Cleaning and Processing):
(Abhijeet Pisal,Rohit Pandey )

| **Step** | **Process Description**                                                                 | **Output/Visualization**                                                                   |
|----------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 1        | Import the dataset `Disaster_test.csv` into a DataFrame.                                | Dataset loaded successfully into memory.                                                  |
| 2        | Display the dataset structure using `info()`.                                           | Text output with column names, data types, and non-null counts.                           |
| 3        | Generate descriptive statistics of the dataset using `describe()`.                     | Summary table showing count, mean, std, min, quartiles, and max for numeric columns.      |
| 4        | Count the missing values in each column using `isnull().sum()`.                        | Table displaying the count of missing values for all columns.                             |
| 5        | Fill missing values in `keyword` and `location` columns with 'none'.                   | Missing values in `keyword` and `location` replaced successfully.                         |
| 6        | Verify missing values after filling by re-running `isnull().sum()`.                    | Table showing `0` missing values in the specified columns.                                |
| 7        | Define and apply `clean_text` function to preprocess text (remove URLs, hashtags, etc.). | Cleaned text showing removal of unwanted elements and conversion to lowercase.            |
| 8        | Apply the `clean_text` function to create a new column `cleaned_text`.                 | Sample rows showing the original `text` and its cleaned version in `cleaned_text`.        |
| 9        | Plot text length distribution using `sns.histplot`.                                    | Histogram visualizing the distribution of text lengths (e.g., number of characters/words). |
| 10       | Tokenize text, calculate word frequencies, and identify the 20 most common words.      | Word cloud or table showing the most frequently occurring words.                          |
| 11       | Plot a bar chart for the 20 most common words.                                         | Bar chart displaying words (x-axis) vs. their frequencies (y-axis).                       |
| 12       | Remove stopwords from text using the `remove_stopwords` function.                      | Sample rows showing text before and after stopwords removal.                              |
| 13       | Apply lemmatization to text using the `lemmatize_text` function.                       | Sample rows showing text before and after lemmatization.                                  |

## Training Models :
(Sanket Patil and Rajesh Patil).

This project demonstrates the application of machine learning models to a classification problem using logistic regression and random forest classifiers. Below are the key steps and results:
### 1. Document Vectorization: 
Implements Word2Vec to generate document-level feature vectors.

### 2. **Data Splitting**
The dataset is split into training and testing sets using `train_test_split()` with a test size of 20%. The target variable `y` and the features `X` are used for the split.

### 3. **Logistic Regression**
- Trains and evaluates a logistic regression model with accuracy and classification report metrics.
- A logistic regression model is trained using the training set (`X_train` and `y_train`).
- The model's predictions on the test set (`X_test`) are evaluated using `accuracy_score` and `classification_report`.
- **Accuracy**: 58.96%
- **Classification Report**: 
  - Precision (class 0): 0.59
  - Recall (class 0): 0.98
  - Precision (class 1): 0.68
  - Recall (class 1): 0.07

### 4. **Random Forest Classifier**
- Trains and evaluates a random forest classifier for comparison.
- A random forest classifier is trained on the same training set.
- The predictions on the test set are evaluated with the same metrics.
- **Accuracy**: 70.72%
- **Classification Report**: 
  - Precision (class 0): 0.70
  - Recall (class 0): 0.87
  - Precision (class 1): 0.73
  - Recall (class 1): 0.49

### 5. **Prediction on Test Data**
- The trained logistic regression model is used to predict the target variable on a new dataset (`X_test_x`).
- The predictions are stored in a `DataFrame` and saved to a CSV file.

# Final Prediction by using Random Forest Classifier and downloading csv file:

| **Step** | **Process Description**                                                | **Output/Visualization**                                                           |
|----------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 1        | Initialize a Random Forest Classifier using `RandomForestClassifier()`. | A Random Forest model instance is created.                                        |
| 2        | Train the Random Forest model using the `fit()` method with `X` (features) and `y` (labels). | The model is trained on the provided dataset.                                     |
| 3        | Predict the target variable for the test data (`x_test_w2v`) using `predict()`. | Predictions are generated and stored in `y_pred_r`.                               |
| 4        | Create a submission DataFrame containing `id` and `target` columns.    | A DataFrame is created with columns: `id` from the test data and predictions (`target`). |
| 5        | Save the submission DataFrame as a CSV file named `submission.csv`.    | The file `submission.csv` is saved to the current working directory.              |

