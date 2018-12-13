import random
random.seed(100)
import numpy as np
np.random.seed(100)

from time import time
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils import preProcessWithTokens

def gridSearch(X, y, models, parameters, is_pipeline=True):

    model = Pipeline(models) if is_pipeline else models
    
    grid_search = GridSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')

    print("Performing grid search...")
    print("Parameters %r " %(parameters))
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    return grid_search

def gridSearchAndSave(X, y, searchParams, is_pipeline=True):
    for idx in range(len(searchParams)):
        params = searchParams[idx]
        grid_clf = gridSearch(X, y, params[0], params[1], is_pipeline=is_pipeline)
        # save the model to disk
        pickle.dump(grid_clf, open('gridModelsOld/'+ params[2], 'wb'))
        

def getGridSearchModel(filename):
    return pickle.load(open('gridModelsOld/'+ filename, 'rb'))

class IntegerEncoder:
    def __init__(self, df):
        self.max_word_len = -1
        self.tokenizer = Tokenizer()
        self.df = df

        def updateMaxWordLen(row):
            words = preProcessWithTokens(row['tweet']).split(' ')
            wordLen = len(words)
            self.max_word_len = wordLen if wordLen > self.max_word_len else self.max_word_len
            return ' '.join(words)
        
        self.df['encoded'] = self.df.apply(updateMaxWordLen, axis=1)
        self.tokenizer.fit_on_texts(self.df['encoded'])

def getPaddedInputIntegerSequences(encoder, data):
    intSeqs = encoder.tokenizer.texts_to_sequences(data)
    return sequence.pad_sequences(intSeqs) #maxlen=encoder.max_word_len

labelEncoder = LabelEncoder()
labelEncoder.fit([0, 4])
def getLableEncoder():
    return labelEncoder

def getEncodedClasses(df):
    return labelEncoder.transform(df['sentiment'])

def createLSTMModel(total_words, max_word_length, embedding_size=16, lstm_cell=100):
    model = Sequential()
    model.add(Embedding(total_words, output_dim=embedding_size, input_length=max_word_length))
    model.add(LSTM(lstm_cell, return_sequences=True))
    model.add(LSTM(lstm_cell))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def createCNNModel(total_words, max_word_length, embedding_size=16, dropout_rate=0.3):
    model = Sequential()
    model.add(Embedding(total_words, output_dim=embedding_size, input_length=max_word_length))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(Conv1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

pipelineTFNaiveBayes = [
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
]

parametersTFNaiveBayes = {
    'vect__preprocessor': (preProcessWithTokens,),
    'vect__ngram_range': ((1, 1), (1, 2))  # unigrams or bigrams
}

pipelineTFIDFNaiveBayes = [
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
]

parametersTFIDFNaiveBayes = {
    'vect__preprocessor': (preProcessWithTokens,),
    'vect__ngram_range': ((1, 1), (1, 2))  # unigrams or bigrams
}

pipelineTFLogit = [
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())
]

parametersTFLogit = {
    'vect__preprocessor': (preProcessWithTokens, ),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clf__C': (0.001, 0.01, 0.1, 1.0, 10),
    'clf__random_state': (42, )
}

pipelineTFIDFLogit = [
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression())
]

parametersTFIDFLogit = {
    'vect__preprocessor': (preProcessWithTokens, ),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clf__C': (0.001, 0.01, 0.1, 1.0, 10),
    'clf__random_state': (42, )
}

pipelineTFSVM = [
    ('vect', CountVectorizer()),
    ('clf', LinearSVC()),
]

parametersTFSVM = {
    'vect__preprocessor': (preProcessWithTokens, ),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clf__C': (0.001, 0.01, 0.1, 1.0, 10),
    'clf__random_state': (42,)
}

pipelineTFIDFSVM = [
    ('vect', TfidfVectorizer()),
    ('clf', LinearSVC()),
]

parametersTFIDFSVM = {
    'vect__preprocessor': (preProcessWithTokens, ),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clf__C': (0.001, 0.01, 0.1, 1.0, 10),
    'clf__random_state': (42,)
}

batch_size = [1000]
epochs = [1, 2]
random_state = [42]
embedding_size = [8, 16]
lstm_cell = [50, 100]
parametersLSTM = dict(batch_size=batch_size, epochs=epochs, embedding_size=embedding_size, lstm_cell=lstm_cell)

dropout_rate=[0.3, 0.6]
parametersCNN = dict(batch_size=batch_size, epochs=epochs, embedding_size=embedding_size, dropout_rate=dropout_rate)

naiveBayesTFModelFile = 'naiveBayesTF.grid'
naiveBayesTFIDFModelFile = 'naiveBayesTFIDF.grid'
logitTFModelFile = 'logitTF.grid'
logitTFIDFModelFile = 'logitTFIDF.grid'
svmTFModelFile = 'svmTF.grid'
svmTFIDFModelFile = 'svmTFIDF.grid'
lstmModelFile = 'lstm.grid'
cnnModelFile = 'cnn.grid'

if __name__ == "__main__":
    from utils import loadDataForLearning
    dataFrame = loadDataForLearning()

    searchParamsForPipeline = [ 
        (pipelineTFNaiveBayes, parametersTFNaiveBayes, naiveBayesTFModelFile),
        (pipelineTFIDFNaiveBayes, parametersTFIDFNaiveBayes, naiveBayesTFIDFModelFile),
        (pipelineTFLogit, parametersTFLogit, logitTFModelFile),
        (pipelineTFIDFLogit, parametersTFIDFLogit, logitTFIDFModelFile),
        (pipelineTFSVM, parametersTFSVM, svmTFModelFile),
        (pipelineTFIDFSVM, parametersTFIDFSVM, svmTFIDFModelFile)
    ]
    gridSearchAndSave(dataFrame['tweet'], dataFrame['sentiment'], searchParamsForPipeline)

    encoder = IntegerEncoder(dataFrame)
    totalWords = len(encoder.tokenizer.word_counts)
    parametersLSTM['total_words'] = [totalWords + 1] # +1 as learner is going out of bound  with range [0, totalWords)
    parametersCNN['total_words'] = [totalWords + 1]
    parametersLSTM['max_word_length'] = [encoder.max_word_len]
    parametersCNN['max_word_length'] = [encoder.max_word_len]

    nnSearchParams = [ 
        (KerasClassifier(build_fn=createLSTMModel), parametersLSTM, lstmModelFile),
        (KerasClassifier(build_fn=createCNNModel), parametersCNN, cnnModelFile),
    ]
    
    paddedInput = getPaddedInputIntegerSequences(encoder, dataFrame['encoded'])
    encodedClasses = getEncodedClasses(dataFrame)
    gridSearchAndSave(paddedInput, encodedClasses, nnSearchParams, is_pipeline=False)

