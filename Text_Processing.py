import nltk
import pandas as pd
import numpy as np
import csv
import requests

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def main():
#     return "Flask Page"
# def getData():
#     return requests.get("http://172.26.79.101/")

# if __name__ == "__main__":
#     app.run(debug=True, host = "0.0.0.0", port = 80)

head_score = float
article_score = float
run = 0

def headline_check(headline):
    # for char in headline:
    #     file.replace('"', "")
    global head_score
    headers = pd.read_csv("news_dataset.csv") #dataset with fake and real news headlines
    headers.drop(['Unnamed: 0'], axis = 1, inplace = True) #Clean the dataset

    x = np.array(headers["title"])
    y = np.array(headers["label"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42) #train the neural network
    model = MultinomialNB()
    model.fit(xtrain, ytrain)
    head_score = model.score(xtest, ytest) #R^2 coefficient

    
    data = cv.transform([headline]).toarray()
    return model.predict(data)

def article_check(file):
    # for char in file:
    #     file.replace('"', "")
    global article_score
    articles = pd.read_csv("news_article_dataset.csv") #dataset with fake and real news articles

    x = np.array(articles["text"])
    y = np.array(articles["label"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(xtrain, ytrain)
    article_score = model.score(xtest, ytest) #returns the R^2 coefficient, which represents the correlation of the data (anything above 0.7 is good)

    data = cv.transform([file]).toarray()
    return model.predict(data)

def interpret(heading_file, article_file): #seperate headline and article
    global run
    headline_result = np.array2string(headline_check(heading_file)) #using above function
    for element in headline_result:
        if element == "[" or element == "]" or element == "'": #clean the pandas array output
            headline_result = headline_result.replace(element,"")
    head = headline_result.lower() #standardize

    article_result = np.array2string(article_check(article_file))
    for element in article_result:
        if element == "[" or element == "]" or element == "'":
            article_result = article_result.replace(element,"")
    art = article_result.lower()

    if head == "real" and art == "real": result = "The claims made in this article is reliable."
    if head == "real" and art == "fake": result = "The claims made in this article is not reliable."
    if head == "fake" and art == "real": result = "The claims made in this article are reliable, but the headline might be deceiving"
    if head == "fake" and art == "fake": result = "The claims made in this article aren't reliable"
    run = 1
    return (result + " The reliability score of the title is " + str(head_score) + " and the reliability score of the article is " + str(article_score))

# sentimentAnalyzer = SentimentIntensityAnalyzer()

# def negation_usage(text): #cognitive perspective
#     negations = ["no", "not", "never"]
#     tokens = word_tokenize(text.lower())
#     negationList = [token for token in tokens if token in negations]
#     return negationList.len()

# def pronoun_usage(text): #psychological perspective
#     pronouns = ["i", "i've", "i'd", "he", "she", "him", "her", "you", "yours", "it"]
#     tokens = word_tokenize(text.lower())
#     pronounList = [token for token in tokens if token in pronouns]
#     return pronounList.len()

# def pre_process(text):
#     tokens = word_tokenize(text.lower())
#     lemmatizer = WordNetLemmatizer()
#     # for token in token:
#     #     if token not in stopwords.words('english'):
#     #         filterTokens = [token]
#     filterTokens = [token for token in tokens if token not in stopwords.words('english')]

#     # for token in filterTokens:
#     #     lemmatizeTokens = [(lemmatizer.lemmatize(token))]
#     lemmatizeTokens = [lemmatizer.lemmatize(token) for token in filterTokens]

#     finalText = ' '.join(lemmatizeTokens)
#     return finalText

# def sentiment(text):
#     score = sentimentAnalyzer.polarity_scores(pre_process(text))

#     return score

# def lexical(text):
#     token = ld.hdd(pre_process(text))

#     return token

#print(next(csv))
#print(lexical(text))
#print(sentiment(text))