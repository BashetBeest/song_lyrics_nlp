import pandas as pd
import numpy as np
import torch
import os
import glob
from Title.BackoffLM import BackoffLM
from Title.Evaluator import Evaluator
from Title.TextProcessor import TextProcessor
from Title.TitleModel import TitleModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def main():
    data_df = get_data()
    print(len(data_df))
    final = format_data(data_df)
    print("length:", len(final))
    train, test = split_data(final)
    print("length:", len(train), len(test))
    #print(test.head())
    print("===========================================")

    '''
    Preprocessing - () [], /, /u200b, del before -, remove Lyrics, del after |, *, ?, !, lowercase
     1. Train chance of title being in text - per singer
     2. Train most used length of title
     3. Train title occurrences in text
     '''

    "Initializing..."

    full_lyrics_train = train["Lyric"]
    full_lyrics_test = test["Lyric"]
    y_train = train["Title"]
    y_test = test["Title"]
    artists_train = train["Artist"]

    processor = TextProcessor()
    pro_titles = processor.processTitle(y_train)
    model = TitleModel()

    feature_set = []
    for i in range(len(pro_titles)):
        set = []
        set.append(train["Lyric"][i])
        set.append(train["Artist"][i])
        set.append(pro_titles[i])
        #print(pro_titles[i], ' ===== ', train["Lyric"][i], " ===== ", train["Artist"][i])
        feature_set.append(set)



    test_titles = processor.processTitle(y_test)
    test_set = []
    for i in range(len(test_titles)):
        set = []
        set.append(test["Lyric"][i])
        set.append(test["Artist"][i])
        set.append(test_titles[i])
        #print(test_titles[i], ' ===== ', train["Lyric"][i], " ===== ", train["Artist"][i])
        test_set.append(set)


    #set = model.createFeatureSet(full_lyrics_train, artists_train, pro_titles)

    model.train(test_set)
    model.predict(test_set)



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
    rand = np.random.rand(len(df)) < 0.8
    print(rand)
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

    path = "D:/Documents/Maastricht/Second Year/Period 5/NLP/assignments/Assignment1/nlp2021/Project/song_lyrics_nlp/data/SongLyricsDataset"
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
    return merged

if __name__ == "__main__":
    main()