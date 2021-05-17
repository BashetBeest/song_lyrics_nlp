import pandas as pd
import os
import glob

def main():
    data_df = get_data()
    print(data_df[2].head())

def get_data():
    path = "data/"
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    data = []
    for csv in result:
        data.append(pd.read_csv(csv))
    return data

if __name__ == "__main__":
    main()