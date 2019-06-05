
# mounting google drive
from google.colab import drive
drive.mount('/content/drive/')

# setting up working directory

# %cd /content/drive/My Drive/Data/Instacart

#importing libraries

import pandas as pd
import numpy as np

# reding csv files

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head(2)

print('train shape is ', train.shape)
print('test shape is ', test.shape)

# checking for missing values
print('missing values in train set\n',train.isna().sum())
print('\n\nmissing values in test set\n',test.isna().sum())

"""**Missing value treatment**

There are missing values in target variable 'reordered' which indicates that these products were not reordered.
so filling these missing values with zero.
"""

train['reordered'] = train['reordered'].fillna(value = 0)

# dropping useless column 'order_id'

train.drop(['order_id', 'eval_set', 'department', 'aisle'], axis = 1, inplace = True)
test.drop(['order_id', 'eval_set', 'department', 'aisle'], axis = 1, inplace = True)

#setting index

train = train.set_index(['user_id', 'product_id'])
test = test.set_index(['user_id', 'product_id'])

"""**Importing Tensorflow**"""

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(12)
from sklearn.model_selection import train_test_split

# seprating out the target variable

target = train['reordered']
train_df = train.drop('reordered', axis = 1)

#feature columns
features = train_df.columns

# spliting data in train and validation set

train_x, val_x, train_y, val_y  = train_test_split(train_df, target, test_size = 0.3, random_state = 12)

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('val_x shape is ', val_x.shape)
print('val_y shape is ', val_y.shape)

"""## **Tensorflow Gradient boosted tree model**"""

#preparing features
fc = tf.feature_column
feature_columns = []

for feature_name in features:
  feature_columns.append(fc.numeric_column(feature_name,
                                           dtype=tf.float32))

# creating input function
batch_size = 4096

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
    if shuffle:
      dataset = dataset.shuffle(batch_size)
       
    dataset = (dataset
      .repeat(n_epochs)
      .batch(batch_size)) 
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(train_x, train_y)
eval_input_fn = make_input_fn(val_x, val_y, shuffle=False, n_epochs=1)

params = {
  'n_trees': 50,
  'max_depth': 3,
  'n_batches_per_layer': 1,
  'center_bias': True
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(train_input_fn, max_steps=100)
results = est.evaluate(eval_input_fn)
pd.Series(results).to_frame()

"""Preparing test data for prediction"""

test_input_fn = lambda: tf.data.Dataset.from_tensors(dict(test))

"""Making prediction for test data"""

pred_test = list(est.predict(test_input_fn))
preds = np.array([pred['class_ids'] for pred in pred_test]).ravel()

"""**Preparing data for submission file**"""

sub = pd.read_csv('sample_submission.csv')
sub.head()

del sub, train, train_x, train_y, val_x, val_y
test['prediction'] = preds

test = test.reset_index()
sub_df = test[['user_id', 'product_id', 'prediction']]
del test, preds

# getting test user_id and order_id

orders_df = pd.read_csv('orders.csv')
test_orders = orders_df.loc[orders_df.eval_set=='test',("user_id", "order_id") ]
del orders_df

sub_df = sub_df.merge(test_orders, on='user_id', how='left')
sub_df.head()

# dropping user_id column and converting product id to integer

sub_df.drop('user_id', axis = 1, inplace = True)
sub_df.product_id = sub_df.product_id.astype(int)

# creating dictonary to store order_id and product_id of the products that each order will contain

dic = dict()
for row in sub_df.itertuples():
    if row.prediction== 1:
        try:
            dic[row.order_id] += ' ' + str(row.product_id)
        except:
            dic[row.order_id] = str(row.product_id)

for order in sub_df.order_id:
    if order not in dic:
        dic[order] = 'None'

"""**Creating submission**"""

#Convert the dictionary into a DataFrame
sub_df = pd.DataFrame.from_dict(dic, orient='index')

#Reset index
sub_df.reset_index(inplace=True)
#Set column names
sub_df.columns = ['order_id', 'products']

sub_df.head()

sub_df.to_csv('instacart_submission.csv', index = False)

!jupyter nbconvert --config instacart.py
