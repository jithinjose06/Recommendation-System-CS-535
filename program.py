import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import math

# function to calculate rmse
def get_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# function to calculate average of 2 numbers
def avg_two_numbers(x, y):
    return (x + y) / 2

# function that replaces 0 ratings with the predicted ratings
def imputing_rating(rating,pred):
  return rating if rating>0 else pred


# LOAD THE DATA
train = pd.read_csv('dataset.csv')

# dropping extra column
train = train.drop('Unnamed: 0',axis=1)

df = pd.read_csv('dataset.csv')

# dropping extra column
df = df.drop('Unnamed: 0',axis=1)

# convert data to table
userXitem = pd.pivot_table(df,index = 'item',columns = 'user',values = 'rating').fillna(0)

# convert table back to original format including unrated records
# dev is the test set
dev = userXitem.reset_index().melt('item', value_name='rating')

cols = ['user','item','rating']

dev = dev[cols]

dev['user'] = dev.user.astype(int)


# TRAINING BASELINE MODELS

X_train = train.drop(['rating'], axis='columns')
X_dev = dev.drop(['rating'], axis='columns')
y_train = train['rating'].values
y_dev = dev['rating'].values

# Baseline model 1 - Constant
for prediction in range(1,6):
    y_pred = [prediction]*len(y_dev)
    print(f"RMSE when predicting all ratings as {prediction}:", get_rmse(y_dev, y_pred))

# Baseline model 2 - Avg rating    
avg_rating_across_all_items = np.mean(y_train)
prediction = avg_rating_across_all_items
y_pred = [prediction]*len(y_dev)
print("RMSE when predicting all ratings as the average rating:", get_rmse(y_dev, y_pred))

# Baseline model 3 - Avg user rating
avg_user_ratings_dict = train.groupby(['user'])['rating'].mean().to_dict()
X_train['avg_user_rating'] = X_train['user'].apply(lambda user: avg_user_ratings_dict.get(user, avg_rating_across_all_items))
X_dev['avg_user_rating'] = X_dev['user'].apply(lambda user: avg_user_ratings_dict.get(user, avg_rating_across_all_items))
y_pred = X_dev['avg_user_rating']
print("RMSE when predicting rating as the average user rating:", get_rmse(y_dev, y_pred))

# Baseline model 4 - Avg item rating
avg_item_ratings_dict = train.groupby(['item'])['rating'].mean().to_dict()
X_train['avg_item_rating'] = X_train['item'].apply(lambda item: avg_item_ratings_dict.get(item, avg_rating_across_all_items))
X_dev['avg_item_rating'] = X_dev['item'].apply(lambda item: avg_item_ratings_dict.get(item, avg_rating_across_all_items))
y_pred = X_dev['avg_item_rating']
print("RMSE when predicting rating as the average item rating:", get_rmse(y_dev, y_pred))

# Baseline model 5 - Avg user rating and item rating
X_train['avg_user_and_item_rating'] = X_train[['avg_user_rating', 'avg_item_rating']].apply(lambda x: avg_two_numbers(*x), axis='columns')
X_dev['avg_user_and_item_rating'] = X_dev[['avg_user_rating', 'avg_item_rating']].apply(lambda x: avg_two_numbers(*x), axis='columns')
y_pred = X_dev['avg_user_and_item_rating']
print("RMSE when predicting rating as the mean of the average user and item ratings:", get_rmse(y_dev, y_pred))

# Baseline model 6 - LightGBM few features
import lightgbm as lgb
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_dev)
print("RMSE when predicting using lightgbm and two features:", get_rmse(y_dev, y_pred))

# creating new column to store predicted rating
dev['pred'] = y_pred

# replacing 0 ratings with predicted ratings
dev['final_rating'] = dev[['rating', 'pred']].apply(lambda x: imputing_rating(*x), axis='columns')

dev.drop(['rating','pred'],axis='columns',inplace=True)

dev = dev.round({'final_rating':0})

dev['final_rating'] = dev.final_rating.astype(int)


dev.to_csv('output.csv',header = None,index=False)


# generating final ouput file
open('out.txt', 'w').write('\n'.join(map(' '.join, __import__('csv').reader(open('output.csv')))))

