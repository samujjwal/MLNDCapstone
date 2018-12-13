import random
random.seed(100)
import numpy as np
np.random.seed(100)

from gridSearch import getGridSearchModel, naiveBayesTFModelFile, naiveBayesTFIDFModelFile, logitTFModelFile, logitTFIDFModelFile
from gridSearch import svmTFModelFile, svmTFIDFModelFile, lstmModelFile, cnnModelFile, IntegerEncoder, createLSTMModel, createCNNModel
from gridSearch import getEncodedClasses, getPaddedInputIntegerSequences, getIntegerEncoder, getLableEncoder
from gridSearch import *
from utils import preProcessWithTokens, loadDataForTesting
from sklearn.metrics import accuracy_score

#Dummy methods for unpickling to happen properly
def getLSTMModelCreator():
  return None

def getCNNModelCreator():
  return None

gridModelFiles = {}
gridModelFiles['naiveBayesTF'] = naiveBayesTFModelFile
gridModelFiles['naiveBayesTFIDF'] = naiveBayesTFIDFModelFile
gridModelFiles['logitTF'] = logitTFModelFile
gridModelFiles['logitTFIDF'] = logitTFIDFModelFile
gridModelFiles['svmTF'] = svmTFModelFile
gridModelFiles['svmTFIDF'] = svmTFIDFModelFile
gridModelFiles['lstm'] = lstmModelFile
gridModelFiles['cnn'] = cnnModelFile

models = None
allStats = None

def populateModelsAndStats():
  global models 
  models = dict()
  global allStats 
  allStats = dict()
  for file in gridModelFiles:
    stats = dict()
    grid_result = getGridSearchModel(gridModelFiles[file])
    models[file] = grid_result.best_estimator_
    stats['best_score'] = grid_result.best_score_
    stats['best_params'] = grid_result.best_params_
    stats['stds_train'] = grid_result.cv_results_['std_train_score']
    stats['means_train'] = grid_result.cv_results_['mean_train_score']
    stats['stds'] = grid_result.cv_results_['std_test_score']
    stats['means'] = grid_result.cv_results_['mean_test_score']
    stats['params'] = grid_result.cv_results_['params']
    allStats[file] = stats

def getModels():
  global models
  if models == None:
    populateModelsAndStats()

  return models

def getStats():
  global allStats
  if allStats == None:
    populateModelsAndStats()

  return allStats

def preProcessTweet(dataRow):
  return preProcessWithTokens(dataRow['tweet'])

def testAll(testData, datatype='test'):
  allModels = getModels()
  encoder = getIntegerEncoder()
  processedTweets = testData.apply(preProcessTweet, axis=1)
  paddedInput = getPaddedInputIntegerSequences(encoder, processedTweets)
  encodedClasses = getEncodedClasses(testData)
  labelEncoder = getLableEncoder()
  allPredictions = []
  for key in allModels:
    model = allModels[key]
    if key == 'lstm' or key == 'cnn':
        predictions = model.predict(paddedInput)
        predictions = labelEncoder.inverse_transform(predictions)
    else:
        predictions = model.predict(testData['tweet'])

    allPredictions.append(predictions)
    print ("%s %s accuracy score:  %f " %(key, datatype, accuracy_score(predictions, testData['sentiment'])))
  
  return allPredictions

if __name__ == "__main__":
  statsDetails = getStats()
  for file in statsDetails:
    stats = statsDetails[file]
    print("Best: %f using %s for model %s" % (stats['best_score'], stats['best_params'], file))
    for mean_train, std_train, mean, stdev, param in zip(stats['means_train'], stats['stds_train'], stats['means'], stats['stds'], stats['params']):
        print("Parameters: %r" % (param))
        print("Train Accuracy: %f (%f)" % (mean_train, std_train))
        print("Validation Accuracy: %f (%f)" % (mean, stdev))
        print()