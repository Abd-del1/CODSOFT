
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load the spam sms dataset
data = pd.read_csv(r'C:\Users\Md. Muqtadir\Codsoft\Spam SMS\spam.csv', encoding='latin-1')

#preprocessing the input data
data.drop_duplicates(inplace=True)
data['labels']  = data['v1'].map({'ham': 'ham', 'spam':'spam'})
x= data['v2']
y= data['labels']


#split the data into two sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Create a Tf-IDF 
tfidf_vectorizer = TfidfVectorizer()

#fit the vectorizer to trainig data
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

#Initialize a naive bayes classifier 
classifier = MultinomialNB()

#Train the classifier 
classifier.fit(x_train_tfidf,y_train)

#Transform the test data using same vectorizer 
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Make predictions
y_pred = classifier.predict(x_test_tfidf)

#Calculate accuracy
accuracy= accuracy_score(y_test,y_pred)

#Display classification report 
report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])

#Create a  progress bar
progress_bar= tqdm(total=100, position =0, leave=True)

#Simulate progress update 
for i in range(10,101,10):
    progress_bar.update(10)
    progress_bar.set_description(f"progress: {i}%")

progress_bar.close()

#Display the results
print(f"Accuracy: {accuracy:.2f}")
print('Classification Report:')
print (report)

results = pd.DataFrame({
    'Message': x_test,
    'Actual Label': y_test,
    'Predicted Label': y_pred
})

print(results.head(20))
