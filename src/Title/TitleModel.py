

class TitleModel:

    def __init__(self):
        self.voc = {}
        self.av_len_title = {}
        self.probs = 0
        self.feature_set = []

    def createFeatureSet(self, lyrics, artists, titles):
        for i in range(len(titles)):
            set = []
            set.append(lyrics[i])
            set.append(artists[i])
            set.append(titles[i])
            self.feature_set.append(set)
            # print(self.feature_set[i]," --------")
        return self.feature_set
    
    def train(self, data):
        features = {}
        # set [l, t, a]
        for i in range(len(self.feature_set)):
            lyrics = self.feature_set[i][0]
            artist = self.feature_set[i][1]
            title = self.feature_set[i][2]
            key_list = [artist,title]
            t = tuple(key_list)
            features[t] = self.getOccurr(title, lyrics)


        print(features)


        # [artist] = prob
        # [artist] = av_len_title
        # [artist, title] = occurnces

    def getOccurr(self, title, lyrics):
        num = 0
        lyrics_arr = str(lyrics).split()
        title_arr = str(title).split()

        for i in range(len(lyrics_arr)):
            word_gram = " ".join(lyrics_arr[i:i + len(title_arr)])
            if word_gram == title:
                num += 1
        return num