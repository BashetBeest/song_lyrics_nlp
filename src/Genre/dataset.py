import os
import pathlib
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class MetroLyrics(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data = self.get_metro_lyrics()
        # self.data = self.data[:1000]
        
        ### Preprocessing ###
        self.data.drop_duplicates(subset="lyrics", inplace=True, ignore_index=True)
        self.preprocessing()

        self.genres = self.data["genre"].unique()
        self.genre_to_num = dict((genre, num) for num, genre in enumerate(self.genres))
        self.num_to_genre = dict((num, genre) for num, genre in enumerate(self.genres))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lyrics = self.data["proclyr"][index]
        label = self.data["genre"][index]
        # The label must be numerical
        label = self.genre_to_num[label]
        return {"lyrics": lyrics, "label": label}

    def get_metro_lyrics(self):
        df = pd.read_csv(os.path.join(self.data_dir, "english_cleaned_lyrics.csv"))
        df = df.drop(columns=["Unnamed: 0", "index"])
        return df

    def preprocessing(self):
        """
                INPUT :
                    The Lyrics column
                OUTPUT :
                    Processed lyrics
                    """
        new_data = []
        for i in tqdm(range(len(self.data))):

            lines = self.data["lyrics"][i]
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
            # processed_lines = []
            # for w in tokenized_lines:
            #     if w not in set(stopwords.words('english')):
            #         processed_lines.append(w)
            # processed_lines.append(removed_words)
            final_lines = ' '.join(tokenized_lines)
            new_data.append(final_lines)
        self.data["proclyr"] = new_data # append the preprocessed lyrics
        # print(new_data)
        # return df



class MetroLyricsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir : str, batch_size : int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.collate_fn = None
    
    def setup(self, stage=None):
        if stage is not None: # this is here so setup only happens once
            return
        metro_lyrics = MetroLyrics(self.data_dir)
        self.n_genres = len(metro_lyrics.genre_to_num)
        l = len(metro_lyrics)
        train = int(0.8*l)
        val = int(0.1*l)
        test = l - train - val
        test_train_val_split = [train, val, test]
        self.metro_train, self.metro_val, self.metro_test = random_split(metro_lyrics, test_train_val_split, generator=torch.Generator().manual_seed(69))
        self.define_count_vector()

    def train_dataloader(self):
        return DataLoader(self.metro_train, collate_fn=self.collate_fn, batch_size=self.batch_size) # , num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.metro_val, collate_fn=self.collate_fn, batch_size=self.batch_size) # , num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.metro_test, collate_fn=self.collate_fn, batch_size=self.batch_size) # , num_workers=8)

    def define_count_vector(self):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False, analyzer='word', dtype=np.int16)
        self.vectorizer.fit([x["lyrics"] for x in self.metro_train])
        self.collate_fn = self.make_vector

    def make_vector(self, batch):
        # print(batch)
        # print(batch.shape)
        lyrics = self.vectorizer.transform([batch[0]["lyrics"]])
        for x in batch[1:]:
            lyrics = vstack((lyrics, self.vectorizer.transform([x["lyrics"]])))
        # lyrics = [self.vectorizer.transform([x["lyrics"]]) for x in batch]
        labels = [x["label"] for x in batch]
        return lyrics, labels

    def get_vector_len(self):
        return len(self.vectorizer.vocabulary_)
        


# path = pathlib.Path(__file__).parent.absolute()
# path = os.path.join(path, "../../data/MetroLyrics")
# m = MetroLyricsDataModule(path)
# m.setup()
# m.define_count_vector()
# tr = m.train_dataloader()
# for stuff in tr:
#     x, y = stuff
#     # print(stuff)
#     print(x)
#     print(y)
#     break
# d = tr.dataset
# print(len(d))
# print(d[len(d)-1])
