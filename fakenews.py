import re
import nltk
import string
import joblib
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px

def wordpre(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

#Dataset 1 
d1 = pd.read_csv('Datasets/news.csv')
d1['article'] = d1['title'] + d1['text']
d1 = d1.sample(frac=1).copy()  # Shuffle and ensure it's a copy
d1['label'] = d1['label'].map({'REAL': 1, 'FAKE': 0})  # Convert labels safely
d1 = d1.loc[:, ['article', 'label']].dropna()  # Select necessary columns & drop NaNs
d1['article'] = d1['article'].apply(wordpre)  # Apply text processing

#Dataset 2
d2_true = pd.read_csv('Datasets/True.csv')
d2_fake = pd.read_csv('Datasets/Fake.csv')
d2_true['label'] = 1
d2_fake['label'] = 0
d2 = pd.concat([d2_true,d2_fake])
d2['article'] = d2['title'] + d2['text']
d2.sample(frac = 1) #Shuffle 100%
d2 = d2.loc[:,['article','label']]
d2['article'] = d2['article'].apply(wordpre)

#Dataset 3
d3_real = pd.read_csv('Datasets/politifact_fake.csv')
d3_fake = pd.read_csv('Datasets/politifact_real.csv')
d3_real['label'] = 1
d3_fake['label'] = 0
d3 = pd.concat([d3_real, d3_fake])
d3['article'] = d3['title']
d3.sample(frac = 1) #Shuffle 100%
d3 = d3.loc[:,['article','label']]
d3['article'] = d3['article'].apply(wordpre)

#Dataset 4
d4 = pd.read_csv('Datasets/train.csv')
d4['article'] = d4['title'] + d4['text']
d4.sample(frac = 1) #Shuffle 100%
d4 = d4.loc[:,['article','label']]
d4 = d4.dropna()
d4['article'] = d4['article'].apply(wordpre)

#Dataset 5
d5 = pd.read_csv('Datasets/data.csv')
d5['article'] = d5['Headline'] + d5['Body']
d5['label'] = d5['Label']
d5.sample(frac = 1) #Shuffle 100%
d5 = d5.loc[:,['article','label']]
d5 = d5.dropna()
d5['article'] = d5['article'].apply(wordpre)

#Concat all the 5 datasets 
frames = [d1, d2, d3, d4, d5]
d = pd.concat(frames)
d = d.drop_duplicates()
d = d.dropna()

def create_bar_chart():
    d['label'] = d['label'].astype(str)
    fig = px.histogram(d, x='label', text_auto=True)
    return fig

x_train,x_test,y_train,y_test = train_test_split(d['article'], d['label'], test_size=0.2, random_state=2020)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#As we have seen in Jupyter Notebook , Logistic Regression has the highest accuracy so we will be using it as our model

#LogisticRegression
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

Logisticmodel = pipe.fit(x_train, y_train)
prediction = Logisticmodel.predict(x_test)
print('accuracy: {}%'.format(round(accuracy_score(y_test, prediction)*100, 2)))
Logisticmodel_accuracy = round(accuracy_score(y_test, prediction)*100, 2)

# accuracy: 86.28% of Logistic Regression model

# Save the model as a pickle in a file 
joblib.dump(Logisticmodel, 'model.pkl')