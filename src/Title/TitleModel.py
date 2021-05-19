

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
        # set [l, t, a]
        av_len_arr = []

        for i in range(len(self.feature_set)):
            lyrics = self.feature_set[i][0]
            artist = self.feature_set[i][1]
            title = self.feature_set[i][2]
            key_list = [artist,title]
            t = tuple(key_list)
            print(t)
            self.voc[t] = self.getOccurr(title, lyrics)

            if (i > 0 and artist == self.feature_set[i-1][1]) or i == 0:
                av_len_arr.append(len(title.split()))
            else:
                self.av_len_title[artist] = sum(av_len_arr)/len(av_len_arr)
                av_len_arr = []

            if (title == "thinking bout you"):
                print(lyrics)

        print(self.voc)
        print(self.av_len_title)
        print(self.voc["Ariana Grande","thank you next"])


        # [artist] = prob
        # [artist] = av_len_title
        # [artist, title] = occurunces

    def getOccurr(self, title, lyrics):
        num = 0
        lyrics_arr = str(lyrics).split()
        title_arr = str(title).split()

        #AFTER PROCESSING ------ thinking bout you = thinkin' bout you !!!!!!!!!
        for i in range(len(lyrics_arr)-len(title_arr)+1):
            word_gram = " ".join(lyrics_arr[i:i + len(title_arr)])
            #print(word_gram, " ", title, " = ", word_gram == title)

            if word_gram == title:
                #print("TRUEEEEEEEEEEEEEEEEEEEEEEEEE")

                num += 1
        if num == 0 and len(title_arr) > 1:
            #print("LOWER =====================================", title)
            titleN = ' '.join(title.split(' ')[:-1])

            #print("LOWER =====================================", title)
            for i in range(len(lyrics_arr)):
                word_gram = " ".join(lyrics_arr[i:i + len(title_arr)-1])

                if word_gram == titleN:
                    num += 1
                    #print(word_gram, " ", title, " = ", word_gram == titleN)
                    #print("IMPROVE ++++++++++++++++++++++++++++++++++++++++++++++++++")
                    #print(lyrics)

        print(num)
        return num