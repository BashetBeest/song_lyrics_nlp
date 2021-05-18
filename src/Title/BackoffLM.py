import math
from LanguageModel import LanguageModel


class BackoffLM(LanguageModel):
    pass

    index = 0
    dict_list = []
    token_size = 0

    def train(self, tokens):
        """ Train the langauge model on the list of tokens in tokens"""
        self.dict_list = list({} for i in range(self.size))
        self.token_size = len(tokens)

        for j in range(self.size):
            for i in range(len(tokens)):
                str = " "
                str = str.join(tokens[i:i + self.size - j])
                if str in self.dict_list[j]:
                    self.dict_list[j][str] = self.dict_list[j][str] + 1
                else:
                    self.dict_list[j][str] = 1

        for i in range(len(self.dict_list)):
            print(dict(list(self.dict_list[i].items())[:3]))



    def backoffPrepro(self,ngram):
        ngram_str = ' '.join(ngram)
        if len(ngram) == 1:
            if ngram_str in self.dict_list[self.size-1]:
                return math.log2(pow(0.4,self.size-1) * (self.dict_list[self.size-1][ngram_str] / self.token_size))
            else:
                return math.log2(pow(0.4,self.size-1) * (1 / self.token_size))
        else:
            return self.stupidBackoff(ngram, 1)

    def stupidBackoff(self, ngram, index):
        self.index = index
        ngram_str = ' '.join(ngram)
        ngram_reduced = ngram[:len(ngram)-1]
        ngram_red_str = ' '.join(ngram_reduced)

        if index <= self.size - 1:
            if ngram_str in self.dict_list[index-1]:
                return math.log2(pow(0.4,index-1) * self.dict_list[index-1][ngram_str] / self.dict_list[index][ngram_red_str])
            else:
                ngram_lower = ngram[1:len(ngram)]
                return self.stupidBackoff(ngram_lower, index + 1)
        else:
            return self.backoffPrepro(ngram)
