
## **Accessing data using kaggle api**
"""

!pip install -U -q kaggle
!mkdir -p ~/.kaggle

from google.colab import files
files.upload()

!cp kaggle.json ~/.kaggle/

!kaggle competitions download -c instacart-market-basket-analysis

!ls

#uziping data files
! unzip aisles.csv.zip
! unzip departments.csv.zip
! unzip order_products__train.csv.zip
! unzip orders.csv.zip
! unzip products.csv.zip
! unzip sample_submission.csv.zip
! unzip order_products__prior.csv.zip

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# reading csv files
order_products_prior_df = pd.read_csv('order_products__prior.csv')
orders_df = pd.read_csv('orders.csv')
order_products_train_df = pd.read_csv('order_products__train.csv')
aisles_df = pd.read_csv('aisles.csv')
products_df = pd.read_csv('products.csv')
departments_df = pd.read_csv('departments.csv')

"""## **Data Exploration**"""

orders_df.head(2)

order_products_prior_df.head(2)

order_products_train_df.head(2)

aisles_df.head(2)

products_df.head(2)

departments_df.head(2)

#checking orders placed by weekdays


plt.figure(figsize=(8,6))
sns.countplot(x="order_dow", data=orders_df)
plt.ylabel('Count', fontsize = 10)
plt.xlabel('Day of week', fontsize = 10)
plt.title("Frequency of order by week day", fontsize=12)
plt.show()

#checking orders placed by hours of the day
plt.figure(figsize = (10,6))
sns.countplot(x="order_hour_of_day", data = orders_df)
plt.xlabel('hour of the day', fontsize = 10)
plt.ylabel('count', fontsize = 10)
plt.title("Frequency of order by hours")
plt.show()

# checking the time intervel between the orders
plt.figure(figsize = (15,6))
sns.countplot(x="days_since_prior_order", data = orders_df)
plt.xlabel('days_since_prior_order', fontsize = 10)
plt.ylabel('count', fontsize = 10)
plt.title("Frequency of prior orders by days")
plt.show()

# checking the number of products in each order

#concating the dataset
order_products_all_df = pd.concat([order_products_train_df, order_products_prior_df], axis=0)

grouped = order_products_all_df.groupby('order_id')['add_to_cart_order'].aggregate(max).reset_index()

counts = grouped.add_to_cart_order.value_counts()

plt.figure(figsize = (20,6))
sns.barplot(counts.index, counts.values)
plt.xlabel('Number of products', fontsize = 10)
plt.ylabel('Number of orders', fontsize = 10)
plt.title("Number of products by order")
plt.show()

#let's see top 10 products which are reordered the most
grouped = order_products_all_df.groupby('product_id')['reordered'].aggregate({'Count': 'count'})
grouped = pd.merge(grouped, products_df[['product_id', 'product_name']], how='left', on=['product_id'])
grouped = grouped.sort_values(by='Count', ascending=False)[:10]

plt.figure(figsize = (10,6))
sns.barplot(x = 'product_name', y = 'Count', data = grouped)
plt.xlabel('products', fontsize = 10)
plt.ylabel('Number of orders', fontsize = 10)
plt.title("Top 10 reordered products")
plt.xticks(rotation = 'vertical')
plt.show()

# Lets merge datasets for more exploration

order_products_all_df = pd.merge(order_products_all_df, products_df, on = 'product_id', how = 'left')
order_products_all_df = pd.merge(order_products_all_df, aisles_df, on = 'aisle_id', how = 'left')
order_products_all_df = pd.merge(order_products_all_df, departments_df, on = 'department_id', how = 'left')
order_products_all_df.head()

# Top 10 aisle

Count = order_products_all_df['aisle'].value_counts().head(10)
plt.figure(figsize = (10,6))
sns.barplot(Count.index, Count.values)
plt.xlabel('aisle')
plt.ylabel('Count')
plt.title('Top 10 aisles')
plt.xticks(rotation = 'vertical')
plt.show()

# Now let's find out the top departments

Count = order_products_all_df['department'].value_counts().head(10)

plt.figure(figsize = (10,6))
sns.barplot(Count.index, Count.values)
plt.xlabel('Department')
plt.ylabel('Count')
plt.title('Top 10 Departments')
plt.xticks(rotation = 'vertical')
plt.show()

"""Order_df has a column named eval_set which tells us the particular row belongs to which dataset.
 Let,s find out the count of rows in train, test, and prior sets.
"""

row_count = orders_df.eval_set.value_counts()
plt.figure(figsize = (8,6))
sns.barplot(row_count.index, row_count.values)
plt.xlabel('Evaluation sets')
plt.ylabel('Row count')
plt.title('Row count in each evaluation set')
plt.show()

#lets find out the total number of unique cutomers in each dataset
unique_customers = orders_df.groupby('eval_set')['user_id'].apply(lambda x: len(x.unique()))
print(unique_customers)

"""Insights from the above data exploration:



*   Number of orders are high on saturday and sunday as compared other days.
*   Most of the orders are placed during the day between 8 a.m. and 6 p.m.
*   Orders are usually placed once every week or once every month.
*   Most orders contain 5 or 6 products.
*   Banana is the most reordered product. Also most of the reordered products are organic.
*   Top asiles are fresh fruits and fresh vegetables.
*   Produce is the top department.
*   There are 206209 customers in prior. 131209 in train 75000 in test set.
"""

del order_products_all_df

"""creating product features"""

# Preparing Datasets

products = pd.DataFrame()

# number of times the product sold or ordered
products['orders'] = order_products_prior_df.groupby('product_id').size()

#number of times product reordered
products['reorders'] = order_products_prior_df.groupby('product_id')['reordered'].sum()

#reorder rate
products['reorder_rate'] = products.reorders/products.orders

#average position of the product in the cart
products['avg_pos_cart'] = order_products_prior_df.groupby('product_id')['add_to_cart_order'].mean()

products = products.reset_index()
products.head()

products = products_df.merge(products, on = 'product_id')
products = products.merge(aisles_df, on = 'aisle_id', how = 'left')
products = products.merge(departments_df, on = 'department_id', how = 'left')
products.drop(['product_name', 'aisle_id', 'department_id'], axis = 1, inplace = True)

del departments_df, aisles_df, products_df
products.head()

order_products_prior_df = order_products_prior_df.merge(orders_df, on = 'order_id')

order_products_prior_df.head()

"""Creating user features

1. number of orders
2. average days between orders
3. average number of products in the basket
4. total number of distinct products in the basket.
"""

users = pd.DataFrame()

#total numbers of orders placed by each user
users['total_orders'] = orders_df.groupby('user_id').size()

#average duration between the orders
users['average_duration_btw_orders'] = orders_df.groupby('user_id')['days_since_prior_order'].mean()

#total numbers of products in all orders for each customer
users['total_products_ordered'] = order_products_prior_df.groupby('user_id').size()

#total number of distincet products for each customer
users['num_distinct_products'] = order_products_prior_df.groupby('user_id')['product_id'].apply(set).map(len)


#average number of products in basket
users['avg_products_in_basket'] = users.total_products_ordered/users.total_orders

users = users.reset_index()
users.head()

# User and Product relation

user_products = pd.DataFrame()

#how many times a user bought a product

user_products['product_bought_count'] = order_products_prior_df.groupby(['user_id', 'product_id'])['order_id'].count()
user_products = user_products.reset_index()
user_products.head()

#merging features

#merging custome_product with user dataframe
features_df = user_products.merge(users, on = 'user_id', how = 'left')

#merging train_df with products_df
features_df = features_df.merge(products, on = 'product_id', how = 'left')

#cleaning up memory
del user_products, users, products, order_products_prior_df

features_df.head()

# lets seprate out the data having eval_set = train or test from orders_df

orders = orders_df[((orders_df.eval_set=='train') | (orders_df.eval_set=='test'))]

#extracting only useful columns

orders = orders[['user_id', 'eval_set', 'order_id']]

# merging features to the data

final_df = features_df.merge(orders, on = 'user_id', how = 'left' )

# lets seprate out the train and test order data from orders_df

train_df = final_df[final_df.eval_set == 'train']
test_df = final_df[final_df.eval_set == 'test']

# getting the products along with the target variable 'reordered' from order_product_train_df

train_df = train_df.merge(order_products_train_df[['product_id', 'order_id', 'reordered']], on = ['order_id', 'product_id'], how = 'left')
del order_products_train_df
train_df.head()

print('train shape is ', train_df.shape)
print('test shape is ', test_df.shape)

#train_df.to_csv('train.csv', index = False)
#test_df.to_csv('test.csv', index = False)

print('train_df shape is ', train_df.shape)
print('test_df shape is ', test_df.shape)

# checking for missing values
print('missing values in train set\n',train_df.isna().sum())
print('\n\nmissing values in test set\n',test_df.isna().sum())

"""**Missing value treatment**

There are missing values in target variable 'reordered' which indicates that these products were not reordered. so filling these missing values with zero.
"""

train_df['reordered'] = train_df['reordered'].fillna(value = 0)

# dropping useless column 'order_id'

pd.options.mode.chained_assignment = None
train_df.drop(['order_id', 'eval_set', 'department', 'aisle'], axis = 1, inplace = True)
test_df.drop(['order_id', 'eval_set', 'department', 'aisle'], axis = 1, inplace = True)

#setting index

train_df = train_df.set_index(['user_id', 'product_id'])
test_df = test_df.set_index(['user_id', 'product_id'])

"""**Importing Tensorflow**"""

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(12)
from sklearn.model_selection import train_test_split

# seprating out the target variable

target = train_df['reordered']
train = train_df.drop('reordered', axis = 1)
del train_df

#feature columns
features = train.columns

# spliting data in train and validation set

train_x, val_x, train_y, val_y  = train_test_split(train, target, test_size = 0.3, random_state = 12)

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('val_x shape is ', val_x.shape)
print('val_y shape is ', val_y.shape)

"""### **Tensorflow Gradient boosted tree model**"""

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

#clearing up some space

del train_x, train_y, val_x, val_y, train, target

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

test_input_fn = lambda: tf.data.Dataset.from_tensors(dict(test_df))
del train_input_fn, eval_input_fn

pred_test = list(est.predict(test_input_fn))
preds = np.array([pred['class_ids'] for pred in pred_test]).ravel()

"""**Preparing data for submission file**"""

#taking a look at the sample submission file
sample = pd.read_csv('sample_submission.csv')
sample.head()

del sample
test_df['prediction'] = preds

test_df = test_df.reset_index()
sub_df = test_df[['user_id', 'product_id', 'prediction']]
del test_df, preds, pred_test

# getting test user_id and order_id

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

"""**Creating submission csv**"""

#Convert the dictionary into a DataFrame
sub_df = pd.DataFrame.from_dict(dic, orient='index')

#Reset index
sub_df.reset_index(inplace=True)
#Set column names
sub_df.columns = ['order_id', 'products']

sub_df.head()

sub_df.to_csv('instacart_submission.csv', index = False)
