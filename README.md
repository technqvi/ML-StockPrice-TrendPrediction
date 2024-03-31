# StockPrice-TrendPrediction (Uptrend and Downtrend Label)
This project revolved around 2 main parts 
* Building classification model (XGBoost and MLP Deep Learning) to identify the trend direction(Uptrend and Downtrend) of stock price as below(Main Project).
  * Apply XGBoost and Deep Learing (Multi layer perceptron (MLP)) to classify the trend direction of SET(Stock Exchange of Thailand Index).
  * Create label manually on Chart Price 15 minutes timeframe using [Amibroker Software](https://www.amibroker.com/) by marking trend directions like  uptrend and downtrend as labels based on my trading experience with Techincal Analysis Indicator such as EMA, MACD,SIGNAL and Custom Indicator(Combination between MACD and RSI to determine my own trading logic). 
  * We develop 2 models to make trend prediction separately. 
    * UpTrend Model, There are 2 labels  such as  Uptrend=1 and Non-Uptrend=0.
    * DownTrend Model, There are 2 labels  such as  Downtrend=1 and Non-Downtrend=0.
* Building rules-based system trading to trade SET50 Future using Amibroker software
  * Write trading signal logic (buy/sell) using [AmiBroker Formula Language (AFL)](https://www.amibroker.com/guide/AFL.html)
  * Send buy/sell order to Stock Market.
  
 # [TimeSeries-LSTM-Forecast-PriceMovement 2023](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement)
* It is project to predict  stock trend direction of the future price movement  by LSTM Time-Series.
* It is same purpose as this project(https://github.com/technqvi/ML-StockPrice-TrendPrediction)] but user differet approch.
 
 ## Main flow
 ### Tool, OS and Sofware To Run on Production 
* OS and Database: Windows Server and Microsoft SQL Server Express Edition
* Software Package MT4,AMibroker
* Development Framework and Essential Packages: Python 3.8 on Anaconda Env  Scikit-learn, XGboost, Keras/Tensorflow,Pasdas/Numpy

### ML System Process as the following steps below.
 ![SystemProces](https://github.com/technqvi/ML-StockPrice-TrendPrediction/assets/38780060/4103a692-a991-4513-bfb5-5baa45542263)
 
1. Feed real-time price data from the Stock Market via Data Provider into MT4.
2. Pull data from MT4 to Amibroker by setting DDE Universal Data Pluge-In on Abmibroker.
3. Create some technical analysis indicator values like MA,MACD,RSI,My Custom Indicator  as input features by running a batch job on Amibroker in order to  export it as a CSV file every 15 minutes to a given local path.
4. Load model to predict label output by Job on Window task scheduler and store predicted result(uptrend/downtrend) into SQL Server.
5. Ambiborker retrieves data from SQL Server as the data source to visualize prediction results on the chart pane.

### Rule-Based System Process as the following steps below.
![SystemProces-RuleBase](https://github.com/technqvi/ML-StockPrice-TrendPrediction/assets/38780060/14603d6e-c1d7-4548-b667-5d45fd4f7b6e)
1. Feed real-time price data from the Stock Market via Data Provider into MT4.
2. Pull data from MT4 to Amibroker by setting DDE Universal Data Pluge-In on Abmibroker.
3. Rule-based system that apply technical analysis indicator like RSI,ATR,MACD process buy/sell signal logic and send order to MT-4.
4. MT-4 push buy/sell order to stock market.

## Main Section  (ML system)
### [S50M15_CleanData](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/S50M15_CleanData) : Preparing Data
Prepare stock price data By dropping, transforming, and enriching so that we can import cleaned data (Open, High, Low, Close) into Amibroker. 
### [MarkLable-S50F](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/MarkLable-S50F) :  Labeling Data
Create train/test dataset  as CSV file on Amiboker software as below.
* Create a signal as a label  on Amibroker by labeling the trend direction toward the chart pane as in the screenshot below, there are  2 label files: 1. Uptrend Model 2. Downtrend Model.
* Create  the features by writing AFL Script to generate technical analysis indicators  on Amibroker.
* Merge the label file and feature file to the training dataset.
<img width="1502" alt="Feature_Label-AmiBroker" src="https://github.com/technqvi/ML-StockPrice-TrendPrediction/assets/38780060/2dba064c-19de-4676-b923-a10c0eac715a">


### [Lab-S50F_XGBoost](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/MarkLable-S50F) : Build XGBoost Model (Main Model)
This is a process to develop the XGBoost model to predict uptrend and downtrend. 
#### [S50F-XGB-Tuning](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Lab-S50F_XGBoost/S50F-XGB-Tuning)
* Tune the model to find the optimal hyperparameter to get the best model on GridSearchCV focusing on Learning_Rate , number of Estimators(The number of trees).
* Evaluate the model by accuracy metric on Cross-Validation and Nested cross-validation and apply early stop to prevent overfitting.
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

## Additional Section
### [3TP-Robot-Trading-Amibroker](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/3TP-Robot-Trading-Amibroker) 
Rule-based system developed by AFL Amibroker.
### [Filter_MLPrediction](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Filter_MLPrediction)
To monitor model performance, we will get prediction results from the product to analyze  consecutive trends.

### [ML_Advanced_ParamTuning](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/ML_Advanced_ParamTuning)
* Take Multiple Tuning Value On Cross Validation to do performance analysis to get top 10 accuracy measurement
* Find the Best Feature From TopN Accuracy  Performance Analysis on CV by tuning hyperparameters such as learning_rate,n_estimators
  
### [Filter_TradingZone](https://github.com/technqvi/ML-StockPrice-TrendPrediction/tree/main/Filter_TradingZone)
* Do some research regarding trading strategy on  the trading zone/range to create custom indicators to use as input features for building the model.
* Sample custom indicators: Monitoring Consecutive Trend Direction, Resistance & Support Range.  

### Book Reference
[XGBoost With Python](https://machinelearningmastery.com/xgboost-with-python/) | [machine-learning-with-python](https://machinelearningmastery.com/machine-learning-with-python/) | [deep-learning-with-python](https://machinelearningmastery.com/deep-learning-with-python/) | [better-deep-learnin](https://machinelearningmastery.com/better-deep-learning/)
