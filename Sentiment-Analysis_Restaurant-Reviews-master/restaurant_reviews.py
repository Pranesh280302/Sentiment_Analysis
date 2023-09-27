
# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import re
import pickle

# Load the dataset into dataframe
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t')

# Download stopwords
nltk.download('stopwords')

# Import other stopwords and PortetStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Cleaning the reviews
corpus = [] # corpus is list of words
ps = PorterStemmer()
for i in range(len(df)):

  # cleaning special characters from reviews
  review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['Review'][i])

  # converting each review into lowercase
  review = review.lower()

  # tokenizing the review into words
  review_words = review.split()

  # removing the stop words
  review_words = [word for word in review_words if word not in set(stopwords.words('english'))]

  # stemming the words
  review_words = [ps.stem(word) for word in review_words]

  # joining the stemmed words
  review = ' '.join(review_words)

  # add to corpus
  corpus.append(review)



# Create Bag of Words (convert text to numerical vectors)
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=1500)
X = tf.fit_transform(corpus).toarray()
y = df.iloc[:,1].values


# Create a pickle for TFIDVectorizer
pickle.dump(tf, open('tf-transform.pkl', 'wb'))


# Create test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Model Building
from sklearn.naive_bayes import MultinomialNB

# Instantiate the classifier with best params derived from Hyperparameter Tuning(Refer to Restaurant_Reviews.ipynb file).
classifier = MultinomialNB(alpha=0.30000000000000004)

# Fit the classifier to training dataset
classifier.fit(X_train, y_train)



# Creating a pickle file for the Multinomial Naive Bayes model
pickle.dump(classifier, open('restaurant-sentiment-mnb-model.pkl', 'wb'))





