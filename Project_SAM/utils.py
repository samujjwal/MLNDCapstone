import re
import string
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.utils import shuffle

def loadRawData(filePath):
    nltk.download('stopwords')
    nltk.download('punkt')
    df = pd.read_csv(filePath, encoding='latin-1', header=None)
    df.columns = ['sentiment', 'id', 'timestamp', 'query', 'user', 'tweet']
    return df

def loadRawTrainData():
    return loadRawData("trainingandtestdata/training.1600000.processed.noemoticon.csv")

def removeUnusedColumns(df):
    print ('Removing columns - id, timestamp, query, and user')
    df = df.drop(['id', 'timestamp', 'query', 'user'], axis = 1)
    return df

def loadDataForTesting():
    df = loadRawData("trainingandtestdata/testdata.manual.2009.06.14.csv")
    df = removeUnusedColumns(df)
    return df

def loadDataForLearning():
    df = loadRawTrainData()
    df = removeUnusedColumns(df)
    # df = shuffle(df, random_state=10)
    return df

# remove repetitive characters appearing more than 2 times for each word
def cleanWordWithRepetitiveCharacters(tweet):
    wordLen = len(tweet)
    if wordLen == 0:
        return tweet
    
    tweet = tweet.lower()
    newString = []
    newString.append(tweet[0])
    j = 0
    chRepeat = 1
    for i in range(1, wordLen):
        if (newString[j] == tweet[i]):
            chRepeat = chRepeat + 1
            continue
        if (chRepeat == 2):
            newString.append(newString[j])
            j = j + 1
        
        chRepeat = 1
        newString.append(tweet[i])
        j = j + 1
        
    return ''.join(newString)

urlMatcher = re.compile(r'(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?')
def tokenizeURL(tweet, replaceWith='__URL__'):    
    return urlMatcher.sub(replaceWith, tweet)

hashTagMatcher = re.compile(r'(#\S+)')
def tokenizeHashTag(tweet, replaceWith='__HASHTAGS__'):
    return hashTagMatcher.sub(replaceWith, tweet)

userNameMatcher = re.compile(r'(@\S+)')
def tokenizeUserName(tweet, replaceWith='__USER__'):
    return userNameMatcher.sub(replaceWith, tweet)

htmlEncodedTagsMatcher = re.compile(r'&.{2,4};')
def cleanHtmlTags(tweet):
    return htmlEncodedTagsMatcher.sub('', tweet)

# influenced from https://www.grammar-monster.com/lessons/apostrophes_replace_letters.htm
apostrohedWordDict ={ "aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't":"do not", "hadn't":"had not", "hasn't":"has not",
"haven't":"have not", "he'll": "he will", "he's": "he is", "i'll": "i will", "i'm": "i am", "I've": "I have", "isn't":	"is not", "it's": "it is","let's": "let us",
"mightn't": "might not", "mustn't": "must not", "shan't": "shall not", "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is",
"there's": "there is", "they'll": "they will", "they're": "they are", "they've": "they have", "we're": "we are", "we've": "we have", "weren't": "were not",
"wasn't": "was not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is", "who'll": "who will",
"who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not", "you'll": "you will", "you're": "you are", "you've": "you have"}
apostrohedWordMatcher = re.compile(r'\b(' + '|'.join(apostrohedWordDict.keys()) + r')\b')
def cleanApostrophedWords(tweet):
    return apostrohedWordMatcher.sub(lambda matched: apostrohedWordDict[matched.group()], tweet)

punctToEmptyCharTable = str.maketrans('', '', string.punctuation)
def removePunctuations(tweet):
    words = word_tokenize(tweet)
    noPunctWords = [w.translate(punctToEmptyCharTable) for w in words]
    # remove remaining words that are not alphanumeric
    words = [word for word in noPunctWords if word.isalpha()]
    return ' '.join(words)

stopWords = set(stopwords.words('english'))
def removeStopWordsAndGetArrayOfWords(text):    
    return [w for w in text.split(' ') if not w in stopWords]

porter = PorterStemmer()
def stem(words):
    return [porter.stem(w) for w in words]

def preProcessWithTokens(text):
    text = cleanHtmlTags(text)
    text = tokenizeURL(text)
    text = tokenizeHashTag(text)
    text = tokenizeUserName(text) 
    text = cleanWordWithRepetitiveCharacters(text)
    text = cleanApostrophedWords(text)
    text = removePunctuations(text)
    words = removeStopWordsAndGetArrayOfWords(text)
    words = stem(words)
    return ' '.join(words)


