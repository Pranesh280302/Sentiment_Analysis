from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pickle, re, nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load the trained RandomForestClassifier and TFID vectorizer models from local
ps = PorterStemmer()
sia = SentimentIntensityAnalyzer()
classifier = pickle.load(open('restaurant-sentiment-rf-model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf-model.pkl', 'rb'))
doc2vec_model = pickle.load(open('doc2vec-model.pkl', 'rb'))


def predict_sentiment(sample_review):
    sample_review_cleand = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review_cleand = sample_review_cleand.lower()
    sample_review_words = sample_review_cleand.split()
    sample_review_words = [word for word in sample_review_words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    sample_review_words = [ps.stem(word) for word in sample_review_words]
    sample_review_cleand = ' '.join(sample_review_words)
    sample_df = pd.DataFrame()
    sample_df['Review'] = [sample_review]
    sample_df['Review_cleaned'] = [sample_review_cleand]
    sample_df['sentiments_tmp'] = sample_df["Review"].apply(lambda x: sia.polarity_scores(x))
    sample_df = pd.concat([sample_df.drop(['sentiments_tmp'], axis=1), sample_df['sentiments_tmp'].apply(pd.Series)],
                          axis=1)
    sample_df["nb_chars"] = sample_df['Review'].apply(lambda x: len(x))
    sample_df["nb_words"] = sample_df["Review"].apply(lambda x: len(x.split(" ")))
    sample_doc2vec_df = sample_df["Review_cleaned"].apply(lambda x: doc2vec_model.infer_vector(x.split(" "))).apply(pd.Series)
    sample_doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in sample_doc2vec_df.columns]
    sample_df = pd.concat([sample_df, sample_doc2vec_df], axis=1)
    sample_tfidf_result = tfidf.transform(sample_df['Review_cleaned']).toarray()
    sample_tfidf_df = pd.DataFrame(sample_tfidf_result, columns=tfidf.get_feature_names())
    sample_tfidf_df.columns = ["word_" + str(x) for x in sample_tfidf_df.columns]
    sample_tfidf_df.index = sample_df.index
    sample_df = pd.concat([sample_df, sample_tfidf_df], axis=1)
    sample_features = [c for c in sample_df.columns if not c in ['Review', 'Review_cleaned']]

    return classifier.predict(sample_df[sample_features])


app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = predict_sentiment(message)[0]
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
