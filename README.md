# README

## Overview

This repository contains two machine learning projects:
1. **Movie Genre Classification**: Predicting the aggregate rating of a restaurant based on various features.
2. **Spam SMS Detection**: Classifying SMS messages as either spam or not spam.

## Movie Genre Classification

### What the Project Does

This project aims to build a machine learning model to predict the aggregate rating of a restaurant based on various features like location, category, and time of transaction.

### Why the Project is Useful

Understanding and predicting restaurant ratings can help restaurant owners improve their services, attract more customers, and manage their online presence. It also assists customers in making informed dining choices.

### How to Get Started

#### Prerequisites

- Python 3.6 or above
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - tqdm

#### Dataset

Place the dataset files `fraudTrain.csv` and `fraudTest.csv` in the same directory as the script.

#### Running the Script

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required libraries:
   ```sh
   pip install pandas numpy scikit-learn imbalanced-learn tqdm
   ```

3. Run the script:
   ```sh
   python movie_genre_classification.py
   ```

#### Script: `movie_genre_classification.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# Load training dataset
train_data = pd.read_csv("C:/Users/Md. Muqtadir/Codsoft/Credit card fraud/fraudTrain.csv")

# Load testing dataset
test_data = pd.read_csv("C:/Users/Md. Muqtadir/Codsoft/Credit card fraud/fraudTest.csv")

# Combine training and testing data for encoding consistency
combined_data = pd.concat((train_data, test_data), axis=0)

# Extract relevant features from the trans_date_trans_time column
def extract_datetime_features(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
    df.drop('trans_date_trans_time', axis=1, inplace=True)
    return df

combined_data = extract_datetime_features(combined_data)

# Drop irrelevant columns
columns_to_drop = ["first", "last", "job", "dob", "trans_num", "street"]
combined_data.drop(columns_to_drop, axis=1, inplace=True)

# Print columns to check their existence
print("Columns in combined data:", combined_data.columns)

# Ensure columns are correctly named and stripped of any whitespace
combined_data.columns = combined_data.columns.str.strip()

# Separate features and target variables
x_combined = combined_data.drop("is_fraud", axis=1)
y_combined = combined_data["is_fraud"]

# Encode the "merchant" and "category"
label_encoder = LabelEncoder()
x_combined["merchant"] = label_encoder.fit_transform(x_combined["merchant"])
x_combined["category"] = label_encoder.fit_transform(x_combined["category"])

# One-hot encode categorical variables
categorical_columns = ['gender', 'city', 'state']
# Check if categorical columns exist in the data
missing_columns = [col for col in categorical_columns if col not in x_combined.columns]
if missing_columns:
    print(f"Error: The following columns are missing from the data: {missing_columns}")
else:
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    x_combined_categorical = onehot_encoder.fit_transform(x_combined[categorical_columns])

    # Standardize the numeric features
    scaler = StandardScaler()
    x_combined_numeric = scaler.fit_transform(x_combined.drop(columns=categorical_columns))

    # Combine one-hot encoded categorical and numeric features
    x_combined_encoded = np.hstack((x_combined_numeric, x_combined_categorical))

    # Split the combined data back into training and test datasets
    x_train = x_combined_encoded[:len(train_data)]
    x_test = x_combined_encoded[len(train_data):]
    y_train = y_combined[:len(train_data)]
    y_test = y_combined[len(train_data):]

    # Address class imbalance using SMOTE
    sm = SMOTE(random_state=42)
    x_resampled, y_resampled = sm.fit_resample(x_train, y_train)

    # Apply Incremental PCA for dimensionality reduction
    n_components = 100
    ipca = IncrementalPCA(n_components=n_components)

    # Apply Incremental PCA to training data with progress bar
    for batch in tqdm(np.array_split(x_resampled, 10), desc="Applying Incremental PCA"):
        ipca.partial_fit(batch)

    # Transform the training and testing data
    x_resampled_pca = ipca.transform(x_resampled)
    x_test_pca = ipca.transform(x_test)

    # Define and train the random forest model
    rf_classifier = RandomForestClassifier(random_state=42)

    # Fit the model
    rf_classifier.fit(x_resampled_pca, y_resampled)

    # Predict using the trained model
    y_pred = rf_classifier.predict(x_test_pca)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display the necessary report
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{report}")
```

## Spam SMS Detection

### What the Project Does

This project aims to classify SMS messages as either spam or legitimate (ham) using machine learning techniques.

### Why the Project is Useful

Automated spam detection helps in filtering out unwanted messages, thereby improving user experience and enhancing security by preventing phishing attacks and other spam-related threats.

### How to Get Started

#### Prerequisites

- Python 3.6 or above
- Required Python libraries:
  - pandas
  - scikit-learn
  - tqdm

#### Dataset

Place the dataset file `spam.csv` in the same directory as the script.

#### Running the Script

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required libraries:
   ```sh
   pip install pandas scikit-learn tqdm
   ```

3. Run the script:
   ```sh
   python spam_sms_detection.py
   ```

#### Script: `spam_sms_detection.py`

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
data = pd.read_csv(r'C:\Users\Md. Muqtadir\Codsoft\Spam SMS\spam.csv', encoding='latin-1')

# Preprocessing the input data
data.drop_duplicates(inplace=True)
data['labels'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
x = data['v2']
y = data['labels']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Tf-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

# Initialize a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(x_train_tfidf, y_train)

# Transform the test data using the same vectorizer
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Make predictions
y_pred = classifier.predict(x_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display classification report
report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])

# Create a progress bar
progress_bar = tqdm(total=100, position=0, leave=True)

# Simulate progress update
for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f"Progress: {i}%")

progress_bar.close()

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print('Classification Report:')
print(report)
```
