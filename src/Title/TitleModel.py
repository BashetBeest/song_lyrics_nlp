import operator

class TitleModel:

    def __init__(self):
        self.voc = {}
        self.av_len_title = {}
        self.probs = {}
        self.feature_set = []

    def createFeatureSet(self, lyrics, artists, titles):
        for i in range(len(titles)):
            set = []
            set.append(lyrics[i])
            set.append(artists[i])
            set.append(titles[i])
            self.feature_set.append(set)

        return self.feature_set
    
    def train(self, feature_set):
        # set [l, a, t]
        av_len_arr = []
        probs = []

        for i in range(len(feature_set)):

            lyrics = feature_set[i][0]
            artist = feature_set[i][1]
            title = feature_set[i][2]
            key_list = [artist,title]
            t = tuple(key_list)
            #print(t)
            occur = self.getOccurr(title, lyrics)

            if self.voc.get(t) == None:
                #print(self.voc.get(t))
                self.voc[t] = occur
                # if (title == "pon de replay"):
                #     print(self.voc[t])
                #     print(lyrics)


            if (i > 0 and artist == feature_set[i-1][1]) or i == 0:
                av_len_arr.append(len(title.split()))
                if i == len(feature_set)-1:
                    self.av_len_title[feature_set[i - 1][1]] = sum(av_len_arr) / len(av_len_arr)
                    self.probs[feature_set[i - 1][1]] = sum(probs) / len(probs)
                #print(artist," ARTIST ==================")
                if occur > 0:
                    probs.append(1)
                else:
                    probs.append(0)
            else:
                self.av_len_title[feature_set[i-1][1]] = sum(av_len_arr)/len(av_len_arr)
                self.probs[feature_set[i-1][1]] = sum(probs)/len(probs)
                av_len_arr = []
                probs = []



            # if (title == "thinking bout you"):
            #     print(lyrics)
        #
        # print(self.voc)
        # print(self.av_len_title)
        # print(self.probs)


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

        #print(num)

        '''for i in range(len(lyric.split)):
            for j in range av_len-1 : av_len+1
                if word[i:i+av_len+j] in temp_voc:
                    temp_voc++
            
        
        '''
        return num

    def predict(self, feature_set):
        self.correct = 0
        self.part_correct = 0
        self.total = len(feature_set)
        print(len(feature_set))
        for i in range(len(feature_set)):
            lyrics = feature_set[i][0]
            artist = feature_set[i][1]
            title = feature_set[i][2]
            self.predictTitle(lyrics, artist, title)

        for val in self.voc.values():
            if val == 0:
                self.total -= 1
        print("Total: ", len(feature_set)," Total possible predictions: ",self.total)
        print("Correctly labled: ",self.correct)
        print("Partialy labled: ", self.part_correct)

    def predictTitle(self, lyrics,  artist, label):

        if type(lyrics) == str:
            lyrics = lyrics.split(' ')
            lyrics_size = len(lyrics)

            av_title = round(self.av_len_title[artist])
            dict_list = {}
            w = 1


            for j in range(1,5):
                w += 400
                for i in range(lyrics_size):
                    stra = " ".join(lyrics[i:i + j])

                    if stra in dict_list:
                        dict_list[stra] = dict_list[stra] + w
                    else:
                        dict_list[stra] = 1

            high_count = max(dict_list.values())

            title = ""
            #print(artist, ' + ', dict_list, ' ---------------------------------------------------------------------')
            for key, value in dict_list.items():
                if high_count == value:
                    title = key

            #print("Predicted: ",title," REAL: ",label, " COUNT: ", high_count)
            if title == label:
                self.correct += 1
            else:
                add = False
                title_arr = title.split()
                label_arr = label.split()
                for word in title_arr:
                    for word2 in label_arr:
                        if word == word2:
                            add = True
                if add:
                    self.part_correct += 1
            return title
        else:
            return " no lyrics "

