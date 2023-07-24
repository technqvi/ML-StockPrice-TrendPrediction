# StockPrice-TrendPrediction (Uptrend and Downtrend Label)
This project involvs buildig classification model (XGBoost and MLP Deep Learning) to identifies trend direction(uptrend/downtrend) of stock price as the following details.
* Apply XGBoost and Deep Learing (Multi layer perceptron (MLP)) to classify trend direction of SET(Stock Exchange of Thailand Index).
* Create labe manually on Chart Price 15 minutes timeframe using [Amibroker Software](https://www.amibroker.com/) by ploting uptrend and downtrend as label based on my trading experience with Techincal Analysis Indicator such as EMA, MACD,SIGNAL and Custom Indicator(Combination between MACD and RSI to determine my own trading logic). 
* We develop 2 models to make trend predction seperately. 
  * UpTrend Model, There 2 labels  such as  Uptrend=1 and Non-Uptrend=0.
  * DownTrend Model, There 2 labels  such as  Downtrend=1 and Non-Downtrend=0.
 
 ## Main flow
 ### Tool , OS and Sofware To Run on Production 
* OS and Database: Window Server and Microsoft SQL Server Express Edition
* Software Package MT4,AMibroker
* Development Framework and Essential Packages: Python 3.8 on Anaconda Env  Scikit-learn, XGboost, Keras/Tensorflow ,Pasdas/Numpy

### Step&Process
 ![2023-07-14_23-00-44](https://github.com/technqvi/ML-StockPrice-TrendPrediction/assets/38780060/397339a1-9c66-4fee-82d7-167ab79e8e4d)
  

## Code Section in Each Folder 
### [S50M15_CleanData](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/S50M15_CleanData) : Preparing Data
Prepare stock price data By dropping, transforming, and enriching so that we can import cleaned data (Open, High, Low, Close) into Amibroker. 
### [MarkLable-S50F](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/MarkLable-S50F) :  Labeling Data
Create train/test dataset  as CSV file on Amiboker software as below.
* Create signal as label  on Amibroker by labeling trend direction toward chart pane as the screenshot below, there are  2 label files: 1. Uptrend Model 2. Downtrend Model.
* Create  the features by writing AFL Script to generate technical analysis indicators  on Amibroker.
* Merge the label file and feature file to the training dataset.
<img width="1502" alt="Feature_Label-AmiBroker" src="https://github.com/technqvi/ML-StockPrice-TrendPrediction/assets/38780060/7a02e090-7f93-4cdb-9a54-e47765274681">

### [Lab-S50F_XGBoost](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/MarkLable-S50F) : Build XGBoost Model (Main Model)
This is process to develop the XGBoost model to predict uptrend and downtrend. 
#### [S50F-XGB-Tuning](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Lab-S50F_XGBoost/S50F-XGB-Tuning)
* Tune the model to find the optical hyperparameter to get the best model on GridSearchCV focusing on Learning_Rate nEstimators(The number of trees).
* Evaluate the model by accuracy metric on Cross-Validation and Nested Cross Validation and apply early stop to prevent overfitting.
* Perform feature selection as 2 approaches
  * Select features from the model by using model.feature_importances_(Basic Approach).
  * Select feature Cross-Validation Evaluation by finding the sum of feature value score while iterating on cross-validation (Advance Solution).
 #### [S50F-XGB_Model](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Lab-S50F_XGBoost/S50F-XGB_Model)
* Build final model with optimal hyperparameter set resulting from the Tuning process.
* Load the model to make a prediction of unseen data and save the prediction result into the database.
* Analyze & visualize test prediction results to justify model performance.
 
### [Lab-S50F-DNN](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Lab-S5-F-DNN)  : Build MLP-DeepLearing Model(Experiment)
* Build deep learning model on train/test dataset by scaling numeric feature value using StandardScaler and MinMaxScaler (Manual scaling and sklearn.pipeline).
* Tune model on hyperparamter set such as RegularL,Droppout,Batchsize,Epoch to generize model.
* Apply cross-validation and split-validation to evaluate the model's performance.
* Apply the Ensemble Learning technique to improve the model to make better predictions by reducing model variance to get a consistent accuracy score.

### [Filter_MLPrediction](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Filter_MLPrediction)
### [ML_Advanced_ParamTuning](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/ML_Advanced_ParamTuning)

### Book Reference
[XGBoost With Python](https://machinelearningmastery.com/xgboost-with-python/) | [machine-learning-with-python](https://machinelearningmastery.com/machine-learning-with-python/) | [deep-learning-with-python](https://machinelearningmastery.com/deep-learning-with-python/) | [better-deep-learnin](https://machinelearningmastery.com/better-deep-learning/)
