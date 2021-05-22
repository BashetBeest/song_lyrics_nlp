import pandas as pd
import numpy as np
import torch
import os
import glob
from FeatureCreator import FeatureCreator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    data_df = get_data()
    final = format_data(data_df)
    train, test = split_data(final)
    full_lyrics_train = train["Lyric"].values.tolist()
    full_lyrics_test = test["Lyric"].values.tolist()
    y_train = train["Artist"].values.tolist()
    y_test = test["Artist"].values.tolist()
    vectorizer = CountVectorizer(min_df=0, lowercase=False, analyzer='word')
    ind = 0
    for x in full_lyrics_train:
        if str(x) == 'nan':
            del full_lyrics_train[ind]
            del y_train[ind]
        ind+=1
    ind = 0
    print("done")
    for x in full_lyrics_test:
        if str(x) == 'nan':
            del full_lyrics_test[ind]
            del y_test[ind]
        ind+=1
    cleanedList_train = full_lyrics_train
    cleanedList_test = full_lyrics_test
    #cleanedList_train = [x for x in full_lyrics_train if str(x) != 'nan']
    cleanedList_test = [x for x in full_lyrics_test if str(x) != 'nan']
    #print(cleanedList)
    vectorizer.fit(cleanedList_train)

    X_train = vectorizer.transform(cleanedList_train)
    X_test  = vectorizer.transform(cleanedList_test)
    classifier = LogisticRegression()
    print("FIIITTTEEEEDDD")
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("Accuracy:", score)

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
    path = "data/SongLyricsDataset"
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
    '''
    merged=merged.replace(to_replace="Ariana Grande",value="1")
    merged=merged.replace(to_replace="Billie Eilish",value="2")
    merged=merged.replace(to_replace="CardiB",value="4")
    merged=merged.replace(to_replace="Charlie Puth",value="5")
    merged=merged.replace(to_replace="Coldplay",value="6")
    merged=merged.replace(to_replace="Drake",value="7")
    merged=merged.replace(to_replace="Dua Lipa",value="8")
    merged=merged.replace(to_replace="Ed Sheeran",value="0")
    merged=merged.replace(to_replace="Justin Bieber",value="9")
    merged=merged.replace(to_replace="Katy Perry",value="10")
    merged=merged.replace(to_replace="Khalid",value="11")
    merged=merged.replace(to_replace="Lady Gaga",value="12")
    merged=merged.replace(to_replace="Marron 5",value="13")
    merged=merged.replace(to_replace="Nicki Minaj",value="14")
    merged=merged.replace(to_replace="Post Malone",value="15")
    merged=merged.replace(to_replace="Rihanna",value="16")
    merged=merged.replace(to_replace="Selena Gomez",value="17")
    merged=merged.replace(to_replace="Taylor Swift",value="18")
    '''
    return merged

if __name__ == "__main__":
    main()