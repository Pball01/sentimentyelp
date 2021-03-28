from flask import Flask,render_template,url_for,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import json
import nltk # imports the natural language toolkit
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import numpy  as np
import string
import plotly
from nltk.stem import PorterStemmer 
from collections import Counter
from nltk.corpus import stopwords
import string
from langdetect import detect_langs
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
nltk.download('vader_lexicon')
import sklearn.model_selection
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


def load_rows(filepath, nrows = None):
    with open(filepath) as json_file:
        count = 0
        objs = []
        line = json_file.readline()
        while (nrows is None or count < nrows) and line:
            count += 1
            obj = json.loads(line)
            objs.append(obj)
            line = json_file.readline()
        return pd.DataFrame(objs)

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "u": "you",  # added this from the analysis
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english") + list(ENGLISH_STOP_WORDS)+
                   ['will', 'still'])
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Tokenize each word
    text =  nltk.WordPunctTokenizer().tokenize(text)
        
    return text

#using lemmetization
#def lemmatized_words(text):
#    lemm = nltk.stem.WordNetLemmatizer()
#    df2['lemmatized_text'] = list(map(lambda word:
#                                     list(map(lemm.lemmatize, word)),
#                                     df2.Text_Cleaned))


@app.route('/predict',methods=['POST'])
def predict():
    review = load_rows('yelp_academic_dataset_review.json',10000)
    business = load_rows('yelp_academic_dataset_business.json',10000)
    business = business[business['is_open']==1] #selecting only open restaurants
    business = business.drop(['hours','is_open','address', 'postal_code', 'attributes', 'review_count'], axis=1)
    #filter for restaurnats
    business_res = business[business['categories'].str.contains('Restaurants',case=False, na=False)]
    business_res = business_res.rename(columns={'stars': 'business_stars'})
    #removing columns that might not be relevant
    review = review.drop(['useful','user_id','funny', 'cool', 'date'], axis=1)
    review = review.rename(columns={'stars': 'review_stars'})
    final = pd.merge(business_res, review, on='business_id', how='inner')
    
    #selecting stars and text of reviews
    df1 = final[['review_stars', 'text']]

    #We know from past analysis that there is foreign language in this dataset. 
    language = [detect_langs(i) for i in df1.text]
    languages = [str(i[0]).split(':')[0] for i in language]
    df1['language'] = languages
    df1 = df1.loc[df1["language"].isin(["en"])]


    # Create the dictionary 
    label_dictionary ={1 : 1, 2 : 1, 3 : 1,
                        4 : 0, 5 : 0} 

    # Add a new column named Label 
    df1['label'] = df1['review_stars'].map(label_dictionary)

    df2 = df1[['text', 'label']]
    
    df2['Text_Cleaned'] = list(map(clean_text, df2.text))
    #lemmatized_words(df2.Text_Cleaned)
    
    #training and test dataset
    training_data, test_data = sklearn.model_selection.train_test_split(df2, train_size = 0.7, random_state=42)
    #bag of words transforming training and test data
    bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[1,1], lowercase=False)
    X_tr_bow = bow_transform.fit_transform(training_data['Text_Cleaned']) #training data
    X_te_bow = bow_transform.transform(test_data['Text_Cleaned']) #test data

    y_tr = training_data['label']
    y_te = test_data['label']


    #NaiveBayes with bag of words transformation
    #This is the best model
    nb_classifier = MultinomialNB(alpha = 0.9, fit_prior = False)
    nb_classifier.fit(X_tr_bow, y_tr)
    #y_pred = nb_classifier.predict(X_te_bow)
    nb_classifier.score(X_te_bow, y_te)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        clean_data = list(map(clean_text, data))
        vect = bow_transform.transform(clean_data).toarray()
        my_prediction = nb_classifier.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
