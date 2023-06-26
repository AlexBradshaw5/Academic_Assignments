```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
```


```python
df = pd.read_csv('train.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>...</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>T</td>
      <td>Y</td>
      <td>Green</td>
      <td>Triangle</td>
      <td>Snake</td>
      <td>Finland</td>
      <td>...</td>
      <td>2f4cb3d51</td>
      <td>2</td>
      <td>Grandmaster</td>
      <td>Cold</td>
      <td>h</td>
      <td>D</td>
      <td>kr</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>T</td>
      <td>Y</td>
      <td>Green</td>
      <td>Trapezoid</td>
      <td>Hamster</td>
      <td>Russia</td>
      <td>...</td>
      <td>f83c56c21</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Hot</td>
      <td>a</td>
      <td>A</td>
      <td>bF</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Blue</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Russia</td>
      <td>...</td>
      <td>ae6800dd0</td>
      <td>1</td>
      <td>Expert</td>
      <td>Lava Hot</td>
      <td>h</td>
      <td>R</td>
      <td>Jc</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Snake</td>
      <td>Canada</td>
      <td>...</td>
      <td>8270f0d71</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Boiling Hot</td>
      <td>i</td>
      <td>D</td>
      <td>kW</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>N</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Canada</td>
      <td>...</td>
      <td>b164b72a7</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Freezing</td>
      <td>a</td>
      <td>R</td>
      <td>qP</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>299995</th>
      <td>299995</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>T</td>
      <td>N</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Snake</td>
      <td>India</td>
      <td>...</td>
      <td>e027decef</td>
      <td>1</td>
      <td>Contributor</td>
      <td>Freezing</td>
      <td>k</td>
      <td>K</td>
      <td>dh</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299996</th>
      <td>299996</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Green</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Russia</td>
      <td>...</td>
      <td>80f1411c8</td>
      <td>2</td>
      <td>Novice</td>
      <td>Freezing</td>
      <td>h</td>
      <td>W</td>
      <td>MO</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299997</th>
      <td>299997</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Blue</td>
      <td>Star</td>
      <td>Axolotl</td>
      <td>Russia</td>
      <td>...</td>
      <td>314dcc15b</td>
      <td>3</td>
      <td>Novice</td>
      <td>Boiling Hot</td>
      <td>o</td>
      <td>A</td>
      <td>Bn</td>
      <td>7</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299998</th>
      <td>299998</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Green</td>
      <td>Square</td>
      <td>Axolotl</td>
      <td>Costa Rica</td>
      <td>...</td>
      <td>ab0ce192b</td>
      <td>1</td>
      <td>Master</td>
      <td>Boiling Hot</td>
      <td>h</td>
      <td>W</td>
      <td>uJ</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299999</th>
      <td>299999</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Blue</td>
      <td>Trapezoid</td>
      <td>Dog</td>
      <td>Russia</td>
      <td>...</td>
      <td>ad1af2b45</td>
      <td>3</td>
      <td>Contributor</td>
      <td>Freezing</td>
      <td>i</td>
      <td>R</td>
      <td>tP</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>300000 rows × 25 columns</p>
</div>




```python
def split_sets(df, seed=5):
    Y = df.target.values
    X = df.drop(columns=['target'])
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        X, Y, test_size=0.15, random_state=seed)
    x_train_val = x_train_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.20, random_state=seed)
    return  x_train_val, x_train, x_test, x_val, y_train_val, y_train, y_test, y_val

x_train_val, x_train, x_test, x_val, y_train_val, y_train, y_test, y_val \
= split_sets(df, seed=5)
x_train.shape, x_val.shape, x_train_val.shape, x_test.shape
```




    ((204000, 24), (51000, 24), (255000, 24), (45000, 24))




```python
def drop_column(x_train_val, x_train, x_test, x_val, col):
    x_train = x_train.drop(columns=[col])
    x_val = x_val.drop(columns=[col])
    x_train_val = x_train_val.drop(columns=[col])
    x_test = x_test.drop(columns=[col])
    return x_train_val, x_train, x_test, x_val

x_train_val, x_train, x_test, x_val = drop_column(
    x_train_val, x_train, x_test, x_val, 'id')
```


```python
def change_col(x_train_val, x_train, x_test, x_val, col):
    x_train[col] = x_train[col].astype('category')
    x_val[col] = x_val[col].astype('category')
    x_train_val[col] = x_train_val[col].astype('category')
    x_test[col] = x_test[col].astype('category')
    categories = df[col].unique()
    x_train[col] = x_train[col].cat.rename_categories({categories[0]:1, categories[1]:0})
    x_val[col] = x_val[col].cat.rename_categories({categories[0]:1, categories[1]:0})
    x_train_val[col] = x_train_val[col].cat.rename_categories({categories[0]:1, categories[1]:0})
    x_test[col] = x_test[col].cat.rename_categories({categories[0]:1, categories[1]:0})
    return x_train_val, x_train, x_test, x_val

x_train_val, x_train, x_test, x_val = change_col(x_train_val, x_train, x_test, x_val, 'bin_3')
x_train_val, x_train, x_test, x_val = change_col(x_train_val, x_train, x_test, x_val, 'bin_4')
```


```python
def label_encoding_with_UNK(col_train, UNK=True):
    """ Returns a label encoding "UNK" values
    """
    le = LabelEncoder()
    uniq = np.unique(col_train)
    if UNK:
        uniq = np.concatenate((np.array(["UNK"]), uniq))
    le.fit(uniq)
    return le


def transform_column(le, index, x_train, x_val, x_train_val, x_test):
    x_train[index] = le.transform(x_train[index])
    val = [x if x in le.classes_ else 'UNK' for x in x_val[index]]
    x_val[index] = le.transform(val)
    train_val = [x if x in le.classes_ else 'UNK'
                 for x in x_train_val[index]]
    x_train_val[index] = le.transform(train_val)
    test = [x if x in le.classes_ else 'UNK' for x in x_test[index]]
    x_test[index] = le.transform(test)


def hashing_trick(col, n_features=3):
    name = col.name
    col = col.astype('string')
    col_names = [name + "_" + str(i+1) for i in range(n_features)]
    h = FeatureHasher(input_type='string', n_features=n_features)
    out = h.transform(col).toarray()
    return pd.DataFrame(out, columns=col_names)
```


```python
cols_label_encoding = '’nom_0’, ’nom_1’, ’nom_2’, ’nom_3’, ’nom_4’, ’nom_5’, ’nom_6’, \
’nom_7’, ’nom_8’, ’nom_9’, ’ord_1’, ’ord_2’'
cols_feat_hashing = [x.strip(' ') for x in cols_label_encoding.replace('’'," ").split(',')]
cols_label_encoding = ['ord_0', 'ord_3', 'ord_4', 'ord_5']
cols_numerical = ['bin_0', 'bin_1', 'bin_2', 'ord_0', 'day', 'month']
cols_feat_hashing, cols_label_encoding, cols_numerical
```




    (['nom_0',
      'nom_1',
      'nom_2',
      'nom_3',
      'nom_4',
      'nom_5',
      'nom_6',
      'nom_7',
      'nom_8',
      'nom_9',
      'ord_1',
      'ord_2'],
     ['ord_0', 'ord_3', 'ord_4', 'ord_5'],
     ['bin_0', 'bin_1', 'bin_2', 'ord_0', 'day', 'month'])




```python
x_train_hash = x_train[cols_feat_hashing].copy()
x_val_hash = x_val[cols_feat_hashing].copy()
x_test_hash = x_test[cols_feat_hashing].copy()
x_train_hash.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>nom_4</th>
      <th>nom_5</th>
      <th>nom_6</th>
      <th>nom_7</th>
      <th>nom_8</th>
      <th>nom_9</th>
      <th>ord_1</th>
      <th>ord_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>158960</th>
      <td>Red</td>
      <td>Polygon</td>
      <td>Lion</td>
      <td>Finland</td>
      <td>Piano</td>
      <td>dc07effb0</td>
      <td>d173ac7ca</td>
      <td>ec69236eb</td>
      <td>1984d519a</td>
      <td>3b967a668</td>
      <td>Grandmaster</td>
      <td>Freezing</td>
    </tr>
    <tr>
      <th>40740</th>
      <td>Green</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Costa Rica</td>
      <td>Oboe</td>
      <td>fd04a970f</td>
      <td>2df6f79a2</td>
      <td>12b92841d</td>
      <td>19a7677f3</td>
      <td>8b1f75d90</td>
      <td>Novice</td>
      <td>Cold</td>
    </tr>
    <tr>
      <th>8162</th>
      <td>Green</td>
      <td>Triangle</td>
      <td>Lion</td>
      <td>Costa Rica</td>
      <td>Piano</td>
      <td>2cadfed8e</td>
      <td>afebf0803</td>
      <td>370b29add</td>
      <td>f1de422cc</td>
      <td>db8bacb11</td>
      <td>Novice</td>
      <td>Lava Hot</td>
    </tr>
    <tr>
      <th>77267</th>
      <td>Blue</td>
      <td>Trapezoid</td>
      <td>Snake</td>
      <td>Russia</td>
      <td>Oboe</td>
      <td>b97f51ac4</td>
      <td>f497b97d7</td>
      <td>6e0e3ec45</td>
      <td>6da888acf</td>
      <td>b6bb569c7</td>
      <td>Novice</td>
      <td>Freezing</td>
    </tr>
    <tr>
      <th>70843</th>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Canada</td>
      <td>Oboe</td>
      <td>a93b89fc9</td>
      <td>4daee3baf</td>
      <td>479b4bade</td>
      <td>06b7e7cb3</td>
      <td>02a74a666</td>
      <td>Contributor</td>
      <td>Hot</td>
    </tr>
  </tbody>
</table>
</div>




```python
cv = KFold(n_splits=5, random_state=13, shuffle=True)
d = [3, 5, 10, 15, 20]
x_train = pd.concat([x_train, x_val])
x_train_hash = x_train[cols_feat_hashing].copy()
x_test_hash = x_test[cols_feat_hashing].copy()
# delete feature hashing columns from the main dataframes
x_train = x_train.drop(columns=cols_feat_hashing)
x_test = x_test.drop(columns=cols_feat_hashing)
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)

for f in cols_label_encoding:
    le = label_encoding_with_UNK(x_train[f].values, UNK=False)
    transform_column(le, f, x_train, x_val, x_train_val, x_test)
x_train.reset_index(drop=True, inplace=True)
x_val.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train_val = np.concatenate([y_train, y_val], axis = None)
y_train_val[:10]
```




    array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])




```python
avg_auc = {}
for di in d:
    x_train_hashed = pd.concat([hashing_trick(x_train_hash[f], n_features=di)
                                for f in cols_feat_hashing], axis=1)

    x_train_hashed = pd.concat([x_train, x_train_hashed], axis=1)
    
    X = x_train_hashed
    X.reset_index(drop=True, inplace=True)
    Y = y_train_val
    
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        scaler = StandardScaler() # creates the scaler
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = LogisticRegression(random_state=13, C=1).fit(X_train, Y_train)
        y_hat = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(Y_test, y_hat)
        scores.append(auc)
    avg_auc[di] = sum(scores)/len(scores)
avg_auc  
```




    {3: 0.7568530617854234,
     5: 0.7630020906218318,
     10: 0.7652803337916716,
     15: 0.7651298614297974,
     20: 0.7656782724780753}



# 1.4
The Dimension with the Largest AUC was d=20


```python
best_d = 20
x_final_hashed = pd.concat([hashing_trick(x_train_hash[f],n_features=best_d)
                                for f in cols_feat_hashing], axis=1)
x_final = pd.concat([x_train, x_final_hashed], axis=1)


x_test_hashed = pd.concat([hashing_trick(x_test_hash[f],n_features=best_d)
                                for f in cols_feat_hashing], axis=1)
x_test_final = pd.concat([x_test, x_test_hashed], axis=1)

scaler = StandardScaler() # creates the scaler
scaler.fit(x_final)
x_final = scaler.transform(x_final)
x_test_final = scaler.transform(x_test_final)
clf = LogisticRegression(random_state=13, C=1).fit(x_final, y_train_val)
y_hat = clf.predict_proba(x_test_final)[:,1]
roc_auc_score(y_test, y_hat)
```




    0.7656804381005821



# 1.5
Since we are hashing, there is always a chance at collisions. However since our dimension was the largest of our tested dimensions, I think it has the relatively least amount of collisions since it has the most space and therefore my thoughts on selecting this dimension do not change. 
# 2


```python
def reg_target_encoding(train, col, target):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    new_col = f'{col}_mean_enc'
    train[new_col] = np.zeros(train.shape[0])
    global_mean = train.target.mean()
    for train_index, test_index in kfold.split(train, np.zeros(train.shape[0])):
        mean_col = train.loc[train_index].groupby(col).target.mean()
        train[new_col].loc[test_index] = train[col].loc[test_index].map(mean_col)
    train[new_col].fillna(global_mean, inplace=True)
    return pd.DataFrame(train)
```


```python
def mean_encoding_test(test, train, col, target):
    new_col = f'{col}_mean_enc'
    mean_col = train.groupby(col).target.mean()
    global_mean = train.target.mean()
    train[new_col] = train[col].map(mean_col)
    test[new_col] = train[col].map(mean_col)
    train[new_col].fillna(global_mean, inplace=True)
    test[new_col].fillna(global_mean, inplace=True)
    return pd.DataFrame(test), pd.DataFrame(train)
```


```python
indicies = np.random.permutation(len(df))
split_point = len(indicies)*0.8
training_last = df.loc[indicies[:int(split_point)]].reset_index(drop=True)
testing_last = df.loc[indicies[int(split_point):]].reset_index(drop=True)
temp11, temp12 = mean_encoding_test(testing_last, training_last, 'ord_2', 'target') # test, train, col, target
temp2 = reg_target_encoding(training_last, 'ord_2', 'target') # train, col, target
temp31, temp32 = mean_encoding_test(testing_last, training_last, 'nom_2', 'target')
temp4 = reg_target_encoding(training_last, 'nom_2', 'target')
```


```python
temp12.groupby('ord_2').ord_2_mean_enc.value_counts(), temp11.groupby('ord_2').ord_2_mean_enc.value_counts()
```




    (ord_2        ord_2_mean_enc
     Boiling Hot  0.361933           9870
                  0.362337           9847
                  0.361386           9662
                  0.363751           9643
                  0.361360           9590
     Cold         0.258673           5445
                  0.257777           5428
                  0.257036           5424
                  0.257394           5423
                  0.257413           5343
     Freezing     0.226046          16110
                  0.226013          16021
                  0.224603          15866
                  0.224314          15864
                  0.225709          15860
     Hot          0.324862           3647
                  0.326029           3575
                  0.328170           3546
                  0.327147           3535
                  0.324897           3510
     Lava Hot     0.403880          10313
                  0.402143          10256
                  0.404932          10219
                  0.404480          10203
                  0.403650          10146
     Warm         0.288273           3204
                  0.290457           3184
                  0.289106           3143
                  0.292648           3086
                  0.288024           3037
     Name: ord_2_mean_enc, dtype: int64,
     ord_2        ord_2_mean_enc
     Boiling Hot  0.225336          4084
                  0.403817          2461
                  0.362153          2440
                  0.257658          1333
                  0.326222           879
                  0.289702           818
     Cold         0.225336          2247
                  0.403817          1419
                  0.362153          1321
                  0.257658           798
                  0.326222           478
                  0.289702           442
     Freezing     0.225336          6717
                  0.403817          4296
                  0.362153          4018
                  0.257658          2251
                  0.326222          1495
                  0.289702          1318
     Hot          0.225336          1453
                  0.403817           986
                  0.362153           884
                  0.257658           485
                  0.326222           319
                  0.289702           287
     Lava Hot     0.225336          4250
                  0.403817          2685
                  0.362153          2571
                  0.257658          1441
                  0.326222           996
                  0.289702           828
     Warm         0.225336          1344
                  0.403817           885
                  0.362153           804
                  0.257658           447
                  0.326222           280
                  0.289702           240
     Name: ord_2_mean_enc, dtype: int64)




```python
temp2.groupby('ord_2').ord_2_mean_enc.value_counts()
```




    ord_2        ord_2_mean_enc
    Boiling Hot  0.361933           9870
                 0.362337           9847
                 0.361386           9662
                 0.363751           9643
                 0.361360           9590
    Cold         0.258673           5445
                 0.257777           5428
                 0.257036           5424
                 0.257394           5423
                 0.257413           5343
    Freezing     0.226046          16110
                 0.226013          16021
                 0.224603          15866
                 0.224314          15864
                 0.225709          15860
    Hot          0.324862           3647
                 0.326029           3575
                 0.328170           3546
                 0.327147           3535
                 0.324897           3510
    Lava Hot     0.403880          10313
                 0.402143          10256
                 0.404932          10219
                 0.404480          10203
                 0.403650          10146
    Warm         0.288273           3204
                 0.290457           3184
                 0.289106           3143
                 0.292648           3086
                 0.288024           3037
    Name: ord_2_mean_enc, dtype: int64




```python
temp31.groupby('nom_2').nom_2_mean_enc.value_counts(), temp32.groupby('nom_2').nom_2_mean_enc.value_counts()
```




    (nom_2    nom_2_mean_enc
     Axolotl  0.293489          2320
              0.335958          1211
              0.308423          1091
              0.244738           875
              0.318722           874
              0.361210           746
     Cat      0.293489          3351
              0.335958          1667
              0.308423          1527
              0.318722          1253
              0.244738          1250
              0.361210           990
     Dog      0.293489          2583
              0.335958          1205
              0.308423          1172
              0.244738           937
              0.318722           846
              0.361210           722
     Hamster  0.293489          1973
              0.335958           957
              0.308423           945
              0.244738           724
              0.318722           701
              0.361210           583
     Lion     0.293489          6995
              0.335958          3365
              0.308423          3018
              0.244738          2553
              0.318722          2414
              0.361210          1918
     Snake    0.293489          3148
              0.335958          1507
              0.308423          1363
              0.244738          1180
              0.318722          1117
              0.361210           919
     Name: nom_2_mean_enc, dtype: int64,
     nom_2    nom_2_mean_enc
     Axolotl  0.317985           5889
              0.317288           5835
              0.320329           5821
              0.319814           5743
              0.318190           5731
     Cat      0.336594           8031
              0.334112           7934
              0.335036           7914
              0.337526           7899
              0.336522           7843
     Dog      0.245856           6030
              0.245367           6023
              0.242637           6009
              0.245001           5975
              0.244831           5942
     Hamster  0.360801           4818
              0.363627           4766
              0.362939           4700
              0.359560           4692
              0.359138           4628
     Lion     0.292341          16378
              0.293996          16276
              0.294352          16215
              0.293904          16140
              0.292852          16023
     Snake    0.309028           7492
              0.308507           7394
              0.309153           7300
              0.308387           7295
              0.307045           7264
     Name: nom_2_mean_enc, dtype: int64)




```python
temp4.groupby('nom_2').nom_2_mean_enc.value_counts()
```




    nom_2    nom_2_mean_enc
    Axolotl  0.317985           5889
             0.317288           5835
             0.320329           5821
             0.319814           5743
             0.318190           5731
    Cat      0.336594           8031
             0.334112           7934
             0.335036           7914
             0.337526           7899
             0.336522           7843
    Dog      0.245856           6030
             0.245367           6023
             0.242637           6009
             0.245001           5975
             0.244831           5942
    Hamster  0.360801           4818
             0.363627           4766
             0.362939           4700
             0.359560           4692
             0.359138           4628
    Lion     0.292341          16378
             0.293996          16276
             0.294352          16215
             0.293904          16140
             0.292852          16023
    Snake    0.309028           7492
             0.308507           7394
             0.309153           7300
             0.308387           7295
             0.307045           7264
    Name: nom_2_mean_enc, dtype: int64




```python

```
