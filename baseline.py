
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

wordPolarity = None
with open('polarity-lexicon.txt') as f:
    wordPolarity = { key: -1 if val == 'negative' else 1 for key, val in [ line.strip().split(None, 1) for line in f ] }

def getPolarity(word):
    return 0 if word not in wordPolarity else wordPolarity[word]

def baselinePredictor(tweet):
    return sum([ getPolarity(word) for word in tweet.split()])

def predictUsingBaseline(dfRow):
    return 4 if baselinePredictor(dfRow['tweet']) >= 0 else 0

def predict(dataFrame):
    return dataFrame.apply(predictUsingBaseline, axis = 1)

def getConfusionMatrix(df, result):
    return confusion_matrix(df['sentiment'], result)

def getAccuracy(df, result):
    return accuracy_score(df['sentiment'], result)