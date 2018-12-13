# Machine Learning Engineer Nanodegree
**Samujjwal Bhandari**

**Sentiment Analysis from Tweets**

Dataset download link: [http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)

Polarity lexicon for benchmark model: [https://github.com/felipebravom/StaticTwitterSent/blob/master/extra/polarity-lexicon.txt](https://github.com/felipebravom/StaticTwitterSent/blob/master/extra/polarity-lexicon.txt)

The version of python is 3.6 with libraries used

- sklearn version: 0.20.1

- matplotlib version: 3.0.2

- numpy version: 1.15.4

- pandas version: 0.23.4

- nltk version: 3.3

- keras version: 2.2.4

#INSTRUCTIONS

Before running the Exploration.ipynb do the following

1. download dataset [http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) and unzip it.

2. download lexicon data [https://github.com/felipebravom/StaticTwitterSent/blob/master/extra/polarity-lexicon.txt](https://github.com/felipebravom/StaticTwitterSent/blob/master/extra/polarity-lexicon.txt)

3. create folders called gridModels if do not exist and run 'python gridSearch.py'. This will save grid search model inside the gridModels folder.

4. To get the statistics of the different models run 'python testModels.py'

If you want to run grid search with more parameters, create gridModelsOld folder if does not exist and run 'python gridSearch_old.py. To get statistics run 'python oldGridSearchResultProducer.py'