from scipy.sparse import dok_matrix

class TitleModel:

    def __init__(self):
        self.voc = {}
        self.av_len_title = {}
        self.probs = 0
        self.feature_set = []

    def createFeatureSet(self, data):
        for i in range(len(data)):
            set = []
            set.append(data["Lyric"][i])
            set.append(data["Artist"][i])
            set.append(data["Title"][i])
            self.feature_set.append(set)
            # print(self.feature_set[i]," --------")
    
    def train(self, data):
        self.createFeatureSet(data)
        features = dok_matrix((3, len(data)))

        print(len(data))
        print(len(self.feature_set))
        # set [l, t, a]
        for i in range(len(self.feature_set)):
            lyrics = self.feature_set[i][0]
            artist = self.feature_set[i][1]
            title = self.feature_set[i][2]
            features[0, i] = self.getOccurr(title, lyrics)


        print(features)


        # [artist] = prob
        # [artist] = av_len_title
        # [artist, title] = occurnces

    def getOccurr(self, title, lyrics):
        num = 0


        lyrics = str(lyrics).split()
        for word in lyrics:
            print(word, ' ', title)
            if word == title:
                num += 1

        print(num)
        return num