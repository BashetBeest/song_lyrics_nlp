import pandas as pd
import numpy as np
import string
from sklearn.metrics.classification import classification_report
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from  sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn.naive_bayes import MultinomialNB
import warnings


def main():
    data_df = get_data()
    final = format_data(data_df)

    train, test = split_data(final)   

    full_lyrics_train = train["Lyric"].values.tolist()
    full_lyrics_test = test["Lyric"].values.tolist()

    X_train, y_train, X_test, y_test = countVectorizerPrepare(full_lyrics_train, 
                                        full_lyrics_test, train, test)

    runLogisticRegression(X_train, y_train, X_test, y_test)
    runNaiveBayes(X_train, y_train, X_test, y_test)
    runRandomForest(X_train, y_train, X_test, y_test)


    #lyrics_str = " ".join(full_lyrics_train)

    #wordcloud = WordCloud(max_font_size=100, max_words=500, 
    #     background_color="white", width=800, height=400).generate(lyrics_str)
    #plt.figure(figsize=(20,10))
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #wordcloud.to_file("wordcloud_dataset.png")
    #plt.show()

def countVectorizerPrepare(full_lyrics_train, full_lyrics_test, train, test):
    y_train = train["Artist"].values.tolist()
    y_test = test["Artist"].values.tolist()

    full_lyrics_train = [''.join(c for c in s if c not in string.punctuation) 
                                    for s in full_lyrics_train]

    vectorizer = CountVectorizer(lowercase=True, analyzer = 'word', ngram_range = (2,2)) 
    print("vectorizer fitting")
    vectorizer.fit(full_lyrics_train)


    X_train = vectorizer.transform(full_lyrics_train)
    X_test  = vectorizer.transform(full_lyrics_test)

    return X_train, y_train, X_test, y_test

def runNaiveBayes(X_train, y_train, X_test, y_test):
    '''
        Runs a Naive Bayes classifier
        INPUT:
            X_train - train data
            y_train - train label
            X_test - test data
            y_test - test label
    '''
    naive_b = MultinomialNB()
    naive_b.fit(X_train, y_train)
    print("---NAIVE BAYES---")
    predicted = naive_b.predict(X_test)
    print(metrics.classification_report(y_test, predicted))

def runLogisticRegression(X_train, y_train, X_test, y_test):
    '''
        Runs a logistic regression classifier
        INPUT:
            X_train - train data
            y_train - train label
            X_test - test data
            y_test - test label
    '''
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("---LOGISTIC REGRESSION---")
    predicted = classifier.predict(X_test)
    print(metrics.classification_report(y_test, predicted))    

def runRandomForest(X_train, y_train, X_test, y_test):
    '''
        Runs a logistic Random Forest classifier
        INPUT:
            X_train - train data
            y_train - train label
            X_test - test data
            y_test - test label
    '''
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("---Random Forest ---")
    predicted = classifier.predict(X_test)
    print(metrics.classification_report(y_test, predicted)) 
    

def split_data(df, size = 0.8):
    '''
    Split the dataframe into a training and a test set
    INPUT:
        size - size between 0.0 and 1.0 (by default 0.8)
        df - dataframe to split
    OUTPUT:
        train - dataframe containing the train set
        test - dataframe containing the test set
    '''
    np.random.seed(seed=0)
    rand = np.random.rand(len(df)) < size
    train = df[rand]
    test = df[~rand]
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, test


def get_data():
    '''
        OUTPUT :
            data - list of dataframes in the data folder
    '''
    path = "data/"
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    data = []
    for csv in result:
        data.append(pd.read_csv(csv))
    return data

def format_data(csv):
    '''
        INPUT :
            csv - list of dataframes
        OUTPUT :
            final_df - converged dataframes (removes useless columns if any)
    '''
    for df in csv:
        if "Unnamed: 0" in df.columns.to_list():
            del df["Unnamed: 0"]
    merged = pd.concat(csv)
    merged = merged[merged["Lyric"].notna()]
    #merged = merged[(merged["Artist"] == "Drake") | (merged["Artist"] == "Eminem")]
    #merged = merged[(merged["Artist"] != "Drake") & (merged["Artist"] != "Eminem")]
    return merged

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()