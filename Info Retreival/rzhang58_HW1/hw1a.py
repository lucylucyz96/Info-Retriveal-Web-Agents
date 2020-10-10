import argparse
import random 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import re

class EOSClassifier1:
    def train(self, trainX, trainY):
        self.clf = DummyClassifier(strategy='most_frequent')
        self.clf.fit(trainX, trainY)

    def classify(self, testX):
        # same as return ['EOS' for i in range(len(testX))]
        return self.clf.predict(testX)


class EOSClassifier2:
    def train(self, trainX, trainY):
        self.abbrevs = load_wordlist('EOS/abbrevs')

    def classify(self, testX):
        return ['NEOS' if x[3].lower() in self.abbrevs else 'EOS' for x in testX]


class EOSClassifier3:
    def train(self, trainX, trainY):
        self.abbrevs = load_wordlist('EOS/abbrevs')
        self.internal = load_wordlist('EOS/sentence_internal')
        self.time = ['jan','feb','mar','apr','jun','jul','aug','sep','sept','oct','nov','dec','sun','mon','tu','tue',
        'tues','wed','th','thu','thur','thurs','fri','sat','wk','a.m','p.m','cent','bc','b.c','a.d','ad','ce','c.e']
        self.countries = ['u.s',"u.s.a","e.u"]
        self.titles = load_wordlist('EOS/titles')
        self.proper_nouns = load_wordlist('EOS/unlikely_proper_nouns')
        X = [self.preprocess(x) for x in trainX]
        self.clf = LogisticRegression(random_state = 0).fit(X,trainY)

    def preprocess(self, text):
        features = []
        features.append(len(text[3]))
        features.append(len(text[2]))
        features.append(len(text[1]))
        features.append(len(text[5]))
        features.append(len(text[6]))
        features.append(len(text[7]))
        features.append(int(text[8]))
        features.append(int(text[9]))
        features.append(int(text[10]))
        for item in text:
            features.append(1 if item in self.abbrevs else 0)
            features.append(1 if item in self.internal else 0)
            features.append(1 if item in self.time else 0)
            features.append(1 if item in self.countries else 0)
            features.append(1 if item in self.titles else 0)
            features.append(1 if item in self.proper_nouns else 0)
            features.append(1 if re.match('[^A-Za-z0-9]+',item) else 0)
            features.append(1 if re.match('<P>',item) else 0)
            features.append(1 if item.isdigit() else 0)
            features.append(1 if item.isalpha() else 0)
            features.append(1 if re.match('[-;.]', item) else 0)


        return features

    def classify(self, testX):
        '''
        output = []
        for x in testX:
            # if the period is for abbreviation
            if x[3].lower() in self.abbrevs:
                output.append('NEOS')

            # if the period is for internal words
            elif x[3].lower() in self.internal:
                output.append('NEOS')

            # if the period is for time term abbreviations
            elif x[3].lower() in self.time:
                output.append('NEOS')

            # if the period is for title term abbreviations
            elif x[3].lower() in self.titles:
                output.append('NEOS') 

            #if the period is for country term abbreviations
            elif x[3].lower() in self.countries :
                output.append('NEOS')

            else:
                x = self.preprocess(x) 
                x = np.reshape(x,(1,-1))
                output.append(self.clf.predict(x))
        '''
        output = []
        for x in testX:
            x = self.preprocess(x) 
            x = np.reshape(x,(1,-1))
            output.append(self.clf.predict(x))
        return output



def load_wordlist(file):
    with open(file) as fin:
        return set([x.strip() for x in fin.readlines()])


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split()
            X.append(arr[1:])
            y.append(arr[0])
        return X, y


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output')
    return parser.parse_args()


def main():

    args = parseargs()
    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)
    
    #classifier = EOSClassifier1()
    #classifier = EOSClassifier2()
    classifier = EOSClassifier3()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    evaluate(outputs, testY)


if __name__ == '__main__':
    main()