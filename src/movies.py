#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import kagglehub

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import root_mean_squared_error

from tqdm.auto import tqdm

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb


# In[ ]:





# In[2]:


## SOURCE DATA

# Download latest version
# path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
# print("Path to dataset files:", path)


# ### Description 
# ```
# To train an ML model for predicting the movie rating based on input data 
# (like movie title, release year, runtime, revenue, overview, popularity and other features)
# 
# ```

# In[3]:


# !cp  /Users/sethurama/.cache/kagglehub/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/versions/389/TMDB_movie_dataset_v11.csv tmdb-movies.csv


# In[7]:


# Read CSV and analyse 

df = pd.read_csv('tmdb-movies.csv')
df.head()


# In[5]:


df.T


# 

# ## EDA

# In[10]:


df.info()


# In[11]:


df.isna().mean()


# In[12]:


df.shape


# In[13]:


cols_of_interest = ['title','vote_average', 'vote_count', 'status', 'release_date', 'revenue', 'runtime', 'adult', 'budget', 'original_title',
                    'overview', 'popularity', 'genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
cols_of_interest


# In[14]:


df = df[cols_of_interest]


# In[15]:


df.head()


# In[ ]:





# In[16]:


for col in df.columns:
    print(df[col].head(2))
    print(f"N Uniques: {df[col].nunique()}")
    print()


# In[17]:


# => There are a lot of categorical fields with high dimensionality. 


# In[18]:


df.isna().sum()


# In[ ]:





# ## Target Val Analysis

# In[24]:


int(df.vote_average.min()), int(df.vote_average.max())


# In[25]:


mean_rating = df.vote_average.mean()
mean_rating


# In[26]:


df[df.vote_average > 0].vote_average.mean()


# In[27]:


len(df[df.vote_average > 0]) / len(df)


# In[28]:


## Data for only vote_average. 
ndf = df[df.vote_average <= 0]
df = df[df.vote_average > 0]


# ### => Exclude data points where target value is missing.  vote_average 

# In[ ]:





# In[29]:


df.isna().sum()


# In[32]:


### Introduce release_year feature and delete release_date

df['release_year'] = pd.to_datetime(df.release_date).dt.year
mean_year = df.release_year.mean()
df['release_year'] = df.release_year.fillna(mean_year)

del df['release_date']


# ### Prepare data 

# In[33]:


# Fill NA with null


# In[35]:


df['title'] = df.title.fillna('')
df['original_title'] = df.original_title.fillna('')
df['overview'] = df.overview.fillna('')
df['genres'] = df.genres.fillna('')
df['production_companies'] = df.production_companies.fillna('')
df['production_countries'] = df.production_countries.fillna('')
df['spoken_languages'] = df.spoken_languages.fillna('')
df['keywords'] = df.keywords.fillna('')



# In[36]:


df = df.replace(0, np.nan)
df.isna().sum()



# In[37]:


# Impute numericals with mean values 


# In[38]:


for col in ['vote_count', 'runtime', 'budget', 'revenue', 'popularity']:
    col_mean = df[col].mean()
    df[col] = df[col].fillna(col_mean)


# In[39]:


df.isna().sum()


# In[ ]:





# In[40]:


sns.histplot(df.vote_average, bins=50)


# In[ ]:





# In[ ]:





# In[41]:


df.release_year


# In[42]:


df.spoken_languages.value_counts()[:50]


# In[43]:


# Train a model to predict the movie rating (vote_average) for a new movie. 


# In[ ]:





# In[44]:


# Split data.


# In[45]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[46]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train.shape, df_val.shape, df_test.shape


# In[47]:


y_train = df_train['vote_average'].values
y_val = df_val['vote_average'].values
y_test = df_test['vote_average'].values

del df_train['vote_average']
del df_val['vote_average']
del df_test['vote_average']


# In[48]:


df_train.shape, df_val.shape, df_test.shape


# In[ ]:





# In[49]:


# Due to high dimensionality, using FeatureHasher and not using DictVectorizer as it suits high cardinality categorical features.


# In[ ]:





# In[50]:


hasher = FeatureHasher(n_features=25, input_type='dict')


# In[51]:


train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')


# In[52]:


X_train = hasher.fit_transform(train_dicts)
X_val = hasher.transform(val_dicts)


# In[53]:


X_train.shape, X_val.shape


# In[ ]:





# In[ ]:





# ###  Linear regression 

# In[54]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[55]:


y_pred = model.predict(X_val)


# In[ ]:





# In[56]:


rmse = root_mean_squared_error(y_val, y_pred)
rmse


# In[ ]:





# In[57]:


for nf in [10, 25, 50, 100]:
    hasher = FeatureHasher(n_features=nf, input_type='dict')
    
    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    
    X_train = hasher.fit_transform(train_dicts)
    X_val = hasher.transform(val_dicts)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"{nf}_features : rmse({rmse})")


# In[59]:


# For Lasso Regression

for nf in [10, 25, 50, 75, 100]:
    hasher = FeatureHasher(n_features=nf, input_type='dict')
    
    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    
    X_train = hasher.fit_transform(train_dicts)
    X_val = hasher.transform(val_dicts)

    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"{nf}_features : rmse({rmse})")


# ### => n_features for FeatureHasher can be kept 50

# In[60]:


def prep():
    hasher = FeatureHasher(n_features=50, input_type='dict')
    
    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    
    X_train = hasher.fit_transform(train_dicts)
    X_val = hasher.transform(val_dicts)

    return (X_train, X_val)


# In[61]:


## Trees


# In[ ]:





# In[62]:


X_train, X_val = prep()
X_train.shape, X_val.shape, y_train.shape


# In[ ]:





# In[63]:


dtr = DecisionTreeRegressor(
    max_depth=5
)
dtr.fit(X_train, y_train)


# In[64]:


y_pred = dtr.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
rmse


# In[65]:


for d in [3, 5, 7, 10, 15, 20]:
    dtr = DecisionTreeRegressor(
        max_depth=d
    )
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"{d}_depth : rmse({rmse})")


# In[66]:


### Best max_depth for a DT. d = 10


# In[67]:


scores = []
for d in tqdm([5, 10, 15, 20]):
    for s in [10, 20, 50, 100, 200, 500]:
        dtr = DecisionTreeRegressor(
            max_depth=d,
            min_samples_leaf=s
        )
        dtr.fit(X_train, y_train)
        y_pred = dtr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        # print(f"{d}_depth , {s:4d: rmse({rmse})")

        scores.append((d, s, rmse))
        


# In[68]:


cols = ['max_depth', 'min_samples_leaf', 'rmse']
df_scores = pd.DataFrame(scores, columns=cols)
df_scores


# In[ ]:





# In[ ]:





# In[69]:


df_scores_pivot = df_scores.pivot(index='max_depth', columns=['min_samples_leaf'], values=['rmse'])
df_scores_pivot.round(3)


# In[70]:


sns.heatmap(df_scores_pivot, annot=True, fmt='.4f')


# In[71]:


# With depth=15, min_samples_leaf=100 .. RMSE=1.8208


# In[ ]:





# ## Random Forest

# In[ ]:





# In[ ]:





# In[72]:


rfr = RandomForestRegressor(
        random_state=1,
        n_estimators=10,
        max_depth=15,
        min_samples_leaf=100,
        n_jobs=-1
    )
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(rmse)


# In[ ]:





# In[ ]:





# In[73]:


scores = []

for n in tqdm([10, 20, 50, 100]):
    rfr = RandomForestRegressor(
        random_state=1,
        n_estimators=n,
        max_depth=15,
        min_samples_leaf=100,
        n_jobs=-1
    )
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    scores.append((n, rmse))


# In[74]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'rmse'])
plt.plot(df_scores.n_estimators, df_scores.rmse)


# ```
# BEST config for RandomForest:
# 
#         random_state=1,
#         n_estimators=50,
#         max_depth=15,
#         min_samples_leaf=100,
#         n_jobs=-1
# ```
# 

# In[ ]:





# In[ ]:





# ## XGBoost 

# In[ ]:





# In[75]:


X_train_dense = X_train.toarray()
X_val_dense = X_val.toarray()

num_features = X_train_dense.shape[1]
features = [f"f_{i}" for i in range(num_features)]


# In[ ]:





# In[76]:


dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    feature_names=features
)

dval = xgb.DMatrix(
    X_val,
    label=y_val,
    feature_names=features
)


# In[77]:


xgb_params = {
    'eta': 0.5,
    'max_depth': 15,
    'min_child_weight': 100,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,

    'seed': 1,
    'verbosity': 1
}
model = xgb.train(xgb_params, dtrain, num_boost_round=15)  


# In[78]:


y_pred = model.predict(dval)
root_mean_squared_error(y_val, y_pred)


# In[ ]:





# In[79]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.05,\n    'max_depth': 15,\n    'min_child_weight': 100,\n\n    'objective': 'reg:squarederror',\n    'eval_metric': 'rmse',\n\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n    'lambda': 1,      # L2 regularization \n    'alpha': 1,       # L1 regularization \n}\nwatchlist = [(dtrain, 'train'), (dval, 'val')]\nmodel = xgb.train(\n    xgb_params, \n    dtrain, \n    num_boost_round=200,\n    early_stopping_rounds=10, \n    verbose_eval=5,\n    evals=watchlist\n) \n")


# In[80]:


s = output.stdout.strip()
print(s)


# In[81]:


def parse_xgb_output(output):
    s = output.stdout.strip()
    
    lines = s.split('\n')
    xgb_scores = []
    for line in lines:
        r_index, r_train_rmse, r_val_rmse = line.split('\t')
        index = int(r_index.strip('[]'))
        train_rmse = float(r_train_rmse.split(':')[1])
        val_rmse = float(r_val_rmse.split(':')[1])
        xgb_scores.append((index, train_rmse, val_rmse))

    xgb_df = pd.DataFrame(xgb_scores, columns=['iter', 'train_rmse', 'val_rmse'])
    return xgb_df


# In[82]:


xgb_df = parse_xgb_output(output)


# In[83]:


plt.plot(xgb_df.iter, xgb_df.train_rmse, label='train')
plt.plot(xgb_df.iter, xgb_df.val_rmse, label='val')
plt.legend()


# 

# ## Parameter Tuning

# In[ ]:





# In[ ]:





# In[84]:


eta_scores = {} 


# In[95]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 1,\n    'max_depth': 10,\n    'min_child_weight': 10,\n\n    'objective': 'reg:squarederror',\n    'eval_metric': 'rmse',\n\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n    'lambda': 1,      # L2 regularization \n    'alpha': 1,       # L1 regularization \n}\nwatchlist = [(dtrain, 'train'), (dval, 'val')]\nmodel = xgb.train(\n    xgb_params, \n    dtrain, \n    num_boost_round=200,\n    early_stopping_rounds=10, \n    verbose_eval=5,\n    evals=watchlist\n) \n")


# In[96]:


key = f"eta={xgb_params['eta']}"
eta_scores[key] = parse_xgb_output(output)
eta_scores.keys()


# In[97]:


eta_scores['eta=0.05'].head()


# In[98]:


for key, sc_df in eta_scores.items():
    plt.plot(sc_df.iter, sc_df.val_rmse, label=key)
plt.legend()


# ### =>  ETA 0.1 .. Lowers the RMSE quite well

# In[ ]:





# In[100]:


md_scores = {} 


# In[109]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 15,\n    'min_child_weight': 10,\n\n    'objective': 'reg:squarederror',\n    'eval_metric': 'rmse',\n\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n    'lambda': 1,      # L2 regularization \n    'alpha': 1,       # L1 regularization \n}\nwatchlist = [(dtrain, 'train'), (dval, 'val')]\nmodel = xgb.train(\n    xgb_params, \n    dtrain, \n    num_boost_round=200,\n    early_stopping_rounds=10, \n    verbose_eval=5,\n    evals=watchlist\n) \n")


# In[110]:


key = f"max_depth={xgb_params['max_depth']}"
md_scores[key] = parse_xgb_output(output)
md_scores.keys()


# In[111]:


for key, sc_df in md_scores.items():
    plt.plot(sc_df.iter, sc_df.val_rmse, label=key)
plt.legend()


# In[112]:


max_depth = 10 


# In[ ]:





# In[114]:


mcw_scores = {}


# In[128]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 10,\n    'min_child_weight': 5,\n\n    'objective': 'reg:squarederror',\n    'eval_metric': 'rmse',\n\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n    'lambda': 1,      # L2 regularization \n    'alpha': 1,       # L1 regularization \n}\nwatchlist = [(dtrain, 'train'), (dval, 'val')]\nmodel = xgb.train(\n    xgb_params, \n    dtrain, \n    num_boost_round=200,\n    early_stopping_rounds=10, \n    verbose_eval=5,\n    evals=watchlist\n) \n")


# In[129]:


key = f"min_child_weight={xgb_params['min_child_weight']}"
mcw_scores[key] = parse_xgb_output(output)
mcw_scores.keys()


# In[130]:


# mcw_scores.pop('min_child_weight=50')


# In[131]:


for key, sc_df in mcw_scores.items():
    plt.plot(sc_df.iter, sc_df.val_rmse, label=key)
plt.legend()

plt.xlim(20, 200)  # Zoom in on the x-axis from iteration 0 to 100
plt.ylim(1.77, 1.81)  # Zoom in on the y-axis for better visualization of RMSE differences

plt.show()


# In[ ]:


min_child_weight = 100


# In[ ]:





# In[ ]:





# ## Final Model
# 

# In[132]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 100,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
    'lambda': 1,      # L2 regularization 
    'alpha': 1,       # L1 regularization 
}
model = xgb.train(
    xgb_params, 
    dtrain, 
    num_boost_round=200
) 


# In[133]:


y_pred = model.predict(dval)
root_mean_squared_error(y_val, y_pred)


# In[ ]:





# In[134]:


df_full_train = df_full_train.reset_index(drop=True)
df_full_train.head()


# In[135]:


y_full_train = df_full_train.vote_average.values
y_full_train


# In[136]:


del df_full_train['vote_average']
df_full_train.head(2)


# In[137]:


hasher = FeatureHasher(n_features=50, input_type='dict')

full_train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

X_full_train = hasher.fit_transform(full_train_dicts)
X_test = hasher.transform(test_dicts)

X_full_train_dense = X_full_train.toarray()
num_features = X_full_train_dense.shape[1]
features = [f"f_{i}" for i in range(num_features)]


# In[138]:


dfulltrain = xgb.DMatrix(
    X_full_train,
    label=y_full_train,
    feature_names=features
)

dtest = xgb.DMatrix(
    X_test,
    label=y_test,
    feature_names=features
)


# In[139]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 125,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
    'lambda': 1,      # L2 regularization 
    'alpha': 1,       # L1 regularization 
}
model = xgb.train(
    xgb_params, 
    dfulltrain, 
    num_boost_round=200
) 


# In[140]:


y_pred = model.predict(dtest)
root_mean_squared_error(y_test, y_pred)


# In[ ]:





# In[ ]:




