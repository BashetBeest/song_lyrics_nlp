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


def main():
    data_df = get_data()
    print(len(data_df))
    final = format_data(data_df)
    print("length:", len(final))
    train, test = split_data(final)
    print("length:", len(train), len(test))
    print(test.head())
    print("===========================================")

    '''
    Preprocessing - () [], /, /u200b, del before -, remove Lyrics, del after |, *, ?, !, lowercase
     1. Train chance of title being in text - per singer
     2. Train most used length of title
     3. Train title occurrences in text
     '''

    "Initializing..."

    full_lyrics_train = train["Lyric"].values.tolist()
    full_lyrics_test = test["Lyric"].values.tolist()
    y_train = train["Title"].values.tolist()
    y_test = test["Title"].values.tolist()
    artists_train = train["Artist"].values.tolist()

    processor = TextProcessor()
    pro_titles = processor.processTitle(y_train)

    model = TitleModel()
    set = model.createFeatureSet(full_lyrics_train, artists_train, pro_titles)
    model.train(set)


    # print(pro_lyrics)
    # print(pro_titles)



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

    path = "D:/Documents/Maastricht/Second Year/Period 5/NLP/assignments/Assignment1/nlp2021/Project/song_lyrics_nlp/data/"
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