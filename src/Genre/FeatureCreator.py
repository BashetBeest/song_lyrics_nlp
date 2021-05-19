from scipy.sparse import dok_matrix


class FeatureCreator:
    """ Base class for a language model """

    def __init__(self):
        self.voc = {}
        self.id = {}
        self.count = 0

    def createFeatureSet(self, data):

        for (t1, t2, _) in data:
            for w in t1:
                if w in self.voc:
                    self.voc[w] += 1
                else:
                    self.voc[w] = 1
                    self.id[w] = self.count
                    self.count += 1
            for w in t2:
                if w in self.voc:
                    self.voc[w] += 1
                else:
                    self.voc[w] = 1
                    self.id[w] = self.count
                    self.count += 1

    def createFeatures(self, data):
        features = dok_matrix((len(data), self.count))
        # print('created dok matrix',dok_matrix)
        label = []
        for i in range(len(data)):
            for w in data[i][0]:
                if w in self.id:
                    features[i, self.id[w]] = 1
            for w in data[i][1]:
                if w in self.id:
                    features[i, self.id[w]] = 1
            label.append(data[i][2])
        return features, label
