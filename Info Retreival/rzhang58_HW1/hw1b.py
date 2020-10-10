import argparse
from itertools import groupby
import re
import random 

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SegmentClassifier:
    def train(self, trainX, trainY):
        #self.clf = DecisionTreeClassifier()
        self.prev_line = -99
        X = [self.preprocess(x) for x in trainX]
        #self.clf.fit(X, trainY)
        self.svm = SVC(kernel = 'rbf', C = 1) 
        ss = StandardScaler()
        X = ss.fit_transform(X)
        #print(X)
        self.svm.fit(X,trainY)

    def preprocess(self, text):
        features = []
        words = text.split()
        features.append(len(text))
        features.append(len(text.strip()))
        features.append(len(words))

        #count white spaces
        white_space = 0
        for line in text:
            char = list(line)
            for c in char:
                if c == ' ':
                    white_space+=1
        features.append(white_space/len(char))


        #count number of numbers in a line 
        num_digits = len([s for s in words if s.isdigit()])
        features.append(num_digits/len(words))

        #count how many words in line is letter(s) 
        num_english = len([s for s in words if s.isalpha()])
        features.append(num_english/len(words))

        #count the number of special characters (ex.graphics)
        special_char = 0
        for word in words:
            for c in word:
                if(re.match('[^A-Za-z0-9]+',c)):
                    special_char+=1
        features.append(special_char/len(words))

        #count the number of white spaces before the first word

        #count how many words in line is capitalized
        num_cap = sum([sum([c.isupper() for c in word]) for word in words])
        features.append(num_cap/len(words))

        #Matches any line with first word in pattern "word:" for NNHEAD 
        flag_address = 0
        if re.match('^From:|^Article:|^Path:|^Newsgroups:|^Subject:|^Date:|^Organization:|^Lines:|^Approved:|^Message-ID:|^References:', words[0]):
            features.append(1)
            flag_address = 1
        else:
            features.append(0)
        
        #Matches address: email or phone 
        if flag_address == 1:
            features.append(0)
        else:
            address_s= 0
            for word in words:
                if re.match('^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$', word) or re.match('^\D?(\d{3})\D?\D?(\d{3})\D?(\d{4})$', word):
                    address_s +=1
            features.append(address_s)

        #Matches any word following":" or '>' or starting with ":" for quotation 
        score_q= 0
        for word in words:
            if re.match('^(>|:|\s*\S*\s*>|@)',word) or re.match('^.+(wrote|writes|said):',word):
                score_q+=1
        features.append(score_q)

        #matches table 
        if self.prev_line == -99:
            self.prev_line = len(words)
            features.append(0)
        else:
            if self.prev_line-1 < len(words) < self.prev_line+1:
                features.append(1)
            else:
                features.append(0)
            self.prev_line = len(words)


        return features

    def classify(self, testX):
        X = [self.preprocess(x) for x in testX]
        ss= StandardScaler()
        X = ss.fit_transform(X)
        svm_predictions = self.svm.predict(X) 
        return svm_predictions
        #return self.clf.predict(X)
        
def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = ' '.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


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
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    return parser.parse_args()


def main():
    '''
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)
    '''
    args = parseargs()
    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)
    
    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    evaluate(outputs, testY)


if __name__ == '__main__':
    main()