# Numerai
My approach for modeling Numerai data 

This is based on https://numer.ai/ which is a crowdsourced AI hedge fund that provides clean, regularlized and obfuscated financial data for data scientists to submit predictions through machine learning models.

Numerai has its own cryptocurrency called NMR and data scientists are required to stake their own NMRs on the models they submit. You are rewarded with NMRs if the model beats the threshold and in case of bad performance you lose/burn your stake

## Data and Predictions
The data is given out weekly and has 310 regularized features and is divided into train, validation, test and live sets. In addition to this the data is grouped in terms of 'eras' where each era can have thousands of rows. An era represents a month of financial data(this is what Numerai says!).
Each row represents a stock and the same stock can have another row in the same era but there is no way to gather that information as each row id is different.

The training, validation and test data almost never changes and only the live data is the actual new data each week. Also target values for test data are not provided as to avoid overfitting with it. The models are required to submit predictions for test and live datasets. Numerai runs the submitted model on the test data internally.  

The true targets are discrete values -  [0.0. 0.25, 0.50, 0.75, 1.00]   
However the predictions cannot be discrete values and the model is scored on the basis of correlation between true targets and predictions.

## Modeling
Numearai provides several example scripts and helpful documentation to get started. 
There are also various tools and libraries(https://docs.numer.ai/tournament/tools) contributed by fellow data scientists that take care of all the basic stuff so people can fully concentrate on creating their ML models.  

## Approach
I was once ranked as high as #8 out of 1000+ data scientists. Actually I was ranked #3 once(December 2019) but then Numerai used to change their scoring criteria from time to time(in order to find a good metric I guess) and so my #3 isn't recorded in their blockchain. 

Anyways, I wanted to share the approach that helped me in ranking higher on the leaderboard and it was fun learning new things and experimenting with a lot of ML techniques which I otherwise wouldn't have.

I tried basic PCA + XGBoost first. PCA to reduce the data dimensionality (310 features) and chose the feature vectors that explained 90% of the variability. This did not give good results as the dataset is huge (There are 120 eras in training set and each era has approximately 4000-5000 rows). 

There are 310 features and many of the features are correlated(Numerai has infact grouped the columns into 6 categories) but it is highly possible that this correlation amongst features might be different across different eras. This was my hypothesis for going ahead with computing correlation clusters for each era of the training and validation sets. 

The features for each era are clustered and then each era's clusters are compared with every other era's clusters based on few clustering comparison metrics like adjusted_mutual_info_score and adjusted_rand_score. The values range from [0,1] where 1 means 2 clusterings are perfectly similar. So eras whose clusterings exhibit a score greater than a threshold (like 0.7) are deemed to be similar to the era being compared. 

To generate predictions for each test era and the live era, the original PCA + XGBoost pipeline is used on a subset of eras (selected by clustering comparison) 
