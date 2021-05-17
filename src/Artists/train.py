import pandas as pd
import numpy as np
import torch
import os
import glob

def main():
    data_df = get_data()
    print(len(data_df))
    final = format_data(data_df)
    print("length:", len(final))
    train, test = split_data(final)
    print("length:", len(train), len(test))
    print(test.head())

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
    return merged

if __name__ == "__main__":
    main()