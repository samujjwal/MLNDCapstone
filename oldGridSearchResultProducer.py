from gridSearch_old import getGridSearchModel, naiveBayesTFModelFile, naiveBayesTFIDFModelFile, logitTFModelFile, logitTFIDFModelFile
from gridSearch_old import svmTFModelFile, svmTFIDFModelFile, lstmModelFile, cnnModelFile
from utils import preProcessWithTokens

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

allStats = None

def populateModelsAndStats():
  global allStats 
  allStats = dict()
  for file in gridModelFiles:
    stats = dict()
    grid_result = getGridSearchModel(gridModelFiles[file])
    stats['best_score'] = grid_result.best_score_
    stats['best_params'] = grid_result.best_params_
    stats['stds_train'] = grid_result.cv_results_['std_train_score']
    stats['means_train'] = grid_result.cv_results_['mean_train_score']
    stats['stds'] = grid_result.cv_results_['std_test_score']
    stats['means'] = grid_result.cv_results_['mean_test_score']
    stats['params'] = grid_result.cv_results_['params']
    allStats[file] = stats

def getStats():
  global allStats
  if allStats == None:
    populateModelsAndStats()

  return allStats

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