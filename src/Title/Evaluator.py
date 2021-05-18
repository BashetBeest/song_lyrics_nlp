
class Evaluator:

    def __init__(self,lm,prepro,voc):
        self.lm = lm
        self.prepro=prepro
        f = open(voc, encoding="utf-8")
        self.voc = set([l.strip() for l in f.readlines()])
        f.close()

    def eval(self,text,reference):
        text = ["<s>"] * (self.lm.getSize() - 1) + text
        reference = ["<s>"] * (self.lm.getSize() - 1) + reference

        tn=0
        tp=0
        fp=0
        fn=0


        for i in range(self.lm.getSize() - 1,len(text)):
            if text[i] in self.voc and text[i]+"ly" in self.voc:
                #print("Checking",text[i], "with reference:",reference[i]);
                #print(text[i-self.lm.getSize()+1:i+1])
                #print(text[i-self.lm.getSize()+1:i]+[text[i]+"ly"])
                #print(i)
                p1 = self.lm.backoffPrepro(self.prepro.process((text[i-self.lm.getSize()+1:i+1])));
                p2 = self.lm.backoffPrepro(self.prepro.process(text[i-self.lm.getSize()+1:i]+[text[i]+"ly"]));
                #print(p1," ",p2);
                if p1 >= p2:
                    if reference[i] == text[i]:
                        tn+= 1;
                        #print("True negative")
                    else:
                        fn += 1
                        #print("False negative")
                else:
                    if reference[i] == text[i]:
                        fp += 1
                        #print("False positive")
                    else:
                        tp += 1
                        #print("True positive")

        print("True negativ:", tn)
        print("True positiv:", tp)
        print("False negativ:", fn)
        print("False positiv", fp)
        precision = 1.0*tp/(tp+fp)
        recall = 1.0*tp/(tp+fn)
        fs= 2 * (precision * recall )/(precision + recall)
        print("Precision:",precision)
        print("Recall:",recall)
        print("Fscore:",fs)

