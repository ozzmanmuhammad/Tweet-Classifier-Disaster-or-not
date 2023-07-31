# Tweet-Classifier- Disaster or Not Disaster
Multiple experiments to classify tweets into disaster or not disaster tweets using RNN's LSTM, GRU &amp; Bi-LSTM.

## Overview
In this experiment, I tired multiple experiments to build a binary classification model with high accuracy
to classify tweets into "Disaster or Not Disaster tweets". For that I started from traditional Machine learning
baseline model Naive Bayes to complex Bi-Directional LSTM model and also used pretrained 
<a href="https://tfhub.dev/google/universal-sentence-encoder/4" target="_blank">google's Universal Sentence Encoder (USE) model.</a>
  <br/><br/>
Using this I was able to achieve 82% accuracy. All the model that were build were fairly very simple,
with one to two hidden layers and multiple of 8 neurons or units. This was done because of the quantity of the
data with was not much (deep learning require very large amount of data to obtain high accuracy and to avoid overfitting).

## Code and Resources Used
- Python Version: 3.8
- Tensorflow Version: 2.7.0
- Packages: pandas, numpy, sklearn, matplotlib, seaborn
- Editor:  Google Colab

## Dataset and EDA
The dataset was downloaded from the Kaggle: <a href="https://www.kaggle.com/c/nlp-getting-started/data" target="_blank">Natural Language Processing with Disaster Tweets</a>
which contained total of 7,613 tweets, in which 4,342 belonged to Not Disaster class tweets and 3,271 belonged to 
Disaster class tweets. The ratio was fairly balanced (60% Negative Class & 40% positive class) so I considered that Dataset
was not skewed.

dataset examples are following:
                                                        
Class    ::     " Disaster"<br>
Tweet		 ::     Australia's Ashes disaster - how the collapse unfolded at Trent Bridge .<br>

Class    ::     "Not Disaster"<br>
Tweet		 ::     failure is a misfortune but regret is a catastrophe.<br>


## Models Architecture
I have done total of 8 experiments from building baseline model like Naive Bayes, RNN based models,
1D Convolutional NN model, Feed-Forward neural network Dense model and also used pretrained model. I also converted the
text into TextVectorization and also trainable and learnable TextEmbeddings. All of these vectorization and embedding
layers were also build using `tensorflow.keras.layers.experimental.preprocessing.TextVectorization` and
`tensorflow.keras.layers.Embedding` modules. Other details are in the table.

|Models Architecture | Hidden-Layers | 
| ------------- |:-------------------:|
|Naive Bayes|	|
|Feed-forward NN|	Vectorization, Embedding, Pooling1D, Dense|
|LSTM|Vectorization, Embedding, LSTM(64), Dense|
|GRU|	Vectorization, Embedding, GRU(64), Dense|
|Bidirectional-LSTM|	Vectorization, Embedding, Bi-LSTM(64), Dense|
|Conv1D NN|		Vectorization, Embedding, Conv1D(64, 5), Pooling1D, Dense|
|Pretrained USE|	USE layer, Dense(64), Dense|

<br/><br/>

## Model Performance
All of the models were trained on few 5 Epochs because dataset was small and to avoid overfitting, but still
RNN's based models were overfitting. Please check the code file to understant the overfitting. Perfromance details are:

|Models Architecture | Accuracy | 
| ------------- |:-------------------:|
|Naive Bayes|	79%|
|Feed-forward NN|	78%|
|LSTM| 75%|
|GRU|	76%|
|Bidirectional-LSTM|	76%|
|Conv1D NN|		77%|
|Pretrained USE|	82%|

### Other Classification Metrics (Precision, Recall, & f1-score):
<img src="https://github.com/ozzmanmuhammad/ozzmanmuhammad.github.io/blob/main/assets/images/project/tweet_classifier_metrics.jpg" alt="f1score" width="700"/>

#### Models sorted according to their f1-score:
![tweet_classifier_f1](https://github.com/ozzmanmuhammad/Tweet-Classifier-Disaster-or-not/assets/93766242/e028e066-6fbd-4225-9284-0106a4c900f3)




## Predictions & Analysis
I also did custom predictions on unseen tweets from the world, for the I used model 6 which was based on transfer learning universal-sentence-encoder USE model.
It was gave all the correct class of the tweets.<br>

Tweet		      ::      #Beirut declared a “devastated city”, two-week state of emergency officially declared. #LebanonG<br>
Prediction    ::    Real Disaster, Prob: 0.9760<br>

Tweet		      ::     This mobile App may help these people to predict this powerfull M7.9 earthquake <br>
Prediction    ::    Real Disaster, Prob: 0.694<br>

Tweet		      ::      Love the explosion effects in the new spiderman movie <br>
Prediction    ::   Not Real Disaster, Prob: 0.2220<br>
<br>


More experiments can be done to improve models accuracy especially RNN based models which are overfitting with very simple model
so more data is required to avoid overfitting and using some model regularization.
