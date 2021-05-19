import re

import pandas as pd
import numpy as np
import torch
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk

# nltk.download('words')
from tqdm import tqdm

stemmer = PorterStemmer()
from nltk import word_tokenize
from nltk.corpus import stopwords
import os
import glob
from autocorrect import Speller

spell = Speller()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from FeatureCreator import FeatureCreator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    data_df = get_data()
    # print(data_df)
    formatted_data = format_data(data_df)
    # print(formatted_data)
    final = preprocessing(formatted_data)
    # print(final)
    preprocessed_data = preprocessing(final)
    # print(final)

    nan_value = float("NaN")
    preprocessed_data.replace("", nan_value, inplace=True)
    preprocessed_data.dropna(subset=["Lyric"], inplace=True)

    #target class
    Genres = ["Rock", "Pop", "Hip Hop", "Metal", "Country", "Jazz", "Electronic", "R&B"]

    train, test = split_data(preprocessed_data)
    full_lyrics_train = train["Lyric"].values.tolist()
    full_lyrics_test = test["Lyric"].values.tolist()
    y_train = train[Genres].values.tolist()
    y_test = test[Genres].values.tolist()
    vectorizer = CountVectorizer(min_df=0, lowercase=False, analyzer='word')
    ind = 0

    # Visualizing using WordCloud
    # most_used_words_in_lyrics = ' '.join(list(final["Lyric"]))
    # used_lyrics = WordCloud(width=500, height=500).generate(most_used_words_in_lyrics)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(used_lyrics)
    # plt.show()



    # ind = 0
    # for x in full_lyrics_train:
    #     if str(x) == 'nan':
    #         del full_lyrics_train[ind]
    #         del y_train[ind]
    #     ind += 1
    # ind = 0
    # print("done")
    # for x in full_lyrics_test:
    #     if str(x) == 'nan':
    #         del full_lyrics_test[ind]
    #         del y_test[ind]
    #     ind += 1

    cleanedList_train = full_lyrics_train
    cleanedList_test = full_lyrics_test
    cleanedList_train = [x for x in full_lyrics_train if str(x) != 'nan']
    cleanedList_test = [x for x in full_lyrics_test if str(x) != 'nan']
    # print(cleanedList)
    vectorizer.fit(cleanedList_train)

    X_train = vectorizer.transform(cleanedList_train)
    # print(X_train)
    X_test = vectorizer.transform(cleanedList_test)
    # print(X_test)
    classifier = LogisticRegression()
    print("FIIITTTEEEEDDD")
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("Accuracy:", score)


def split_data(df, size=0.8):
    """
    Split the dataframe into a training and a test set
    INPUT:
        size - size between 0.0 and 1.0 (by default 0.8)
        df - dataframe to split
    OUTPUT:
        train - dataframe containing the train set
        test - dataframe containing the test set
    """
    rand = np.random.rand(len(df)) < 0.8
    print(rand)
    train = df[rand]
    test = df[~rand]
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, test


def get_data():
    """
        OUTPUT :
            data - list of dataframes in the data folder
    """
    path = "C:/Users/as/Desktop/song_lyrics_nlp/data"
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    data = []
    for csv in result:
        data.append(pd.read_csv(csv))
    return data


def preprocessing(df):
    """
            INPUT :
                The Lyrics column
            OUTPUT :
                Processed lyrics
                """
    new_data = []
    for i in tqdm(range(df.shape[0])):

        lines = df.iloc[i, 1]
        # print(df.iloc[i, 1])
        # removing non alphabetic characters
        lines = re.sub('[^A-Za-z]', ' ', str(lines))
        # lower every word
        lines = lines.lower()
        # tokenization
        tokenized_lines = word_tokenize(lines)

        # remove non english words
        # words = set(nltk.corpus.words.words())
        # removed_words = " ".join(w for w in nltk.wordpunct_tokenize(lines)
        #                          if w.lower() in words or not w.isalpha())

        # removing stop words,stemming,spell correction
        processed_lines = []
        for w in tokenized_lines:
            if w not in set(stopwords.words('english')):
                processed_lines.append(w)
        # processed_lines.append(removed_words)
        final_lines = ' '.join(processed_lines)
        new_data.append(final_lines)
    # print(new_data)
    return df


def format_data(csv):
    """
        INPUT :
            csv - list of dataframes
        OUTPUT :
            final_df - converged dataframes (removes useless columns if any)
    """
    for df in csv:
        # Remove all unused columns
        if "Unnamed: 0" in df.columns.to_list():
            del df["Unnamed: 0"]
        # if "Artist" in df.columns.to_list():
        #     del df["Artist"]
        if "Title" in df.columns.to_list():
            del df["Title"]
        if "Album" in df.columns.to_list():
            del df["Album"]
        if "Date" in df.columns.to_list():
            del df["Date"]
        if "Year" in df.columns.to_list():
            del df["Year"]
        # # remove stop words
        # stop_words = set(stopwords.words('english'))
        # # tokenization
        # word_tokens = word_tokenize(str(df["Lyric"]))
        # filtered_data = [w for w in word_tokens if not w in stop_words]
        # filtered_data = []
        # for w in word_tokens:
        #     if w not in stop_words:
        #         filtered_data.append(w)

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
