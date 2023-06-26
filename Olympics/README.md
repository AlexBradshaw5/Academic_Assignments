# PCA and Eigenanalysis of Olympic Medal Counts over Time

The file oly_medal_counts.csv contains the total number of Olympic medals won by each country in that year since 1896. In this notebook we will study the trends in the medal counts over time and learn how to extract these trends as features or responses through principal components analysis.

### 0. Getting Started


###  Read in the Data

Read in the data as a pandas dataframe, letting the row names be the country names and the column names be the year of the Olympic games. We will take each country as an individual observation, and take the row corresponding to the $i^{th}$ country as the data vector $x_i,~i=1,...,n$. 

Convert the medal counts to numeric. If there are any NA counts, replace them with 0s.


```python
import pandas as pd
df = pd.read_csv('./oly_medal_counts.csv')
df.fillna(0, inplace=True)
df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>1896</th>
      <th>1900</th>
      <th>1904</th>
      <th>1906</th>
      <th>1908</th>
      <th>1912</th>
      <th>1920</th>
      <th>1924</th>
      <th>1928</th>
      <th>...</th>
      <th>1998</th>
      <th>2000</th>
      <th>2002</th>
      <th>2004</th>
      <th>2006</th>
      <th>2008</th>
      <th>2010</th>
      <th>2012</th>
      <th>2014</th>
      <th>2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>20</td>
      <td>48</td>
      <td>231</td>
      <td>24</td>
      <td>47</td>
      <td>64</td>
      <td>95</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>93</td>
      <td>34</td>
      <td>101</td>
      <td>25</td>
      <td>110</td>
      <td>37</td>
      <td>103</td>
      <td>28</td>
      <td>121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mixed team</td>
      <td>2</td>
      <td>20</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>58</td>
      <td>2</td>
      <td>50</td>
      <td>2</td>
      <td>46</td>
      <td>3</td>
      <td>35</td>
      <td>3</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>3</td>
      <td>9</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>9</td>
      <td>11</td>
      <td>5</td>
      <td>14</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Austria</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>3</td>
      <td>17</td>
      <td>7</td>
      <td>23</td>
      <td>3</td>
      <td>16</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
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
      <th>144</th>
      <td>Guatemala</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Gabon</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Kosovo</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>147</th>
      <td>Fiji</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>148</th>
      <td>Jordan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 36 columns</p>
</div>



### 1. EDA
#### Spaghetti Plot:


```python
import matplotlib.pyplot as plt
plt.figure(figsize=(19,8))
for i in range(len(df)):
    plt.plot(df.iloc[i][1:])
```


    
![png](outputs/output_7_0.png)
    


There is a noticeable shift in the trends starting at 1994. Describe the pattern that you see emerge **starting from 1994 onward**. (2pts)  

A) After 1994, there seems to be larger, consistent spikes for all countries every 4 years and also the year increments are by 2 instead of 4.

#### Why has this change occurred?

A) I would assume this is the year they started adding a more olympic games which would infer more olypmic medals up for grabs and also a winter olympic games which occurs in between the 4 year increments of the original summer one. Upon doing research on the history of the Olympics, 1994 was the first year they had agreed to hold the winter olympics at a different time than the summer ones and continue to increment them every 4 years. 

Let's focus on the trend after 1994 and call this the "recent" era of the Olympics. Subset the data frame to contain years from 1994 onward (including 1994). (1pt)


```python
df[df.columns[24:]]
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994</th>
      <th>1996</th>
      <th>1998</th>
      <th>2000</th>
      <th>2002</th>
      <th>2004</th>
      <th>2006</th>
      <th>2008</th>
      <th>2010</th>
      <th>2012</th>
      <th>2014</th>
      <th>2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>101</td>
      <td>13</td>
      <td>93</td>
      <td>34</td>
      <td>101</td>
      <td>25</td>
      <td>110</td>
      <td>37</td>
      <td>103</td>
      <td>28</td>
      <td>121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>41</td>
      <td>1</td>
      <td>58</td>
      <td>2</td>
      <td>50</td>
      <td>2</td>
      <td>46</td>
      <td>3</td>
      <td>35</td>
      <td>3</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>11</td>
      <td>5</td>
      <td>14</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>3</td>
      <td>17</td>
      <td>3</td>
      <td>17</td>
      <td>7</td>
      <td>23</td>
      <td>3</td>
      <td>16</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>144</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>147</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 12 columns</p>
</div>




```python
df2 = df[df.columns[24:]].copy()
df2.index = df.Country
df2
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994</th>
      <th>1996</th>
      <th>1998</th>
      <th>2000</th>
      <th>2002</th>
      <th>2004</th>
      <th>2006</th>
      <th>2008</th>
      <th>2010</th>
      <th>2012</th>
      <th>2014</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>13</td>
      <td>101</td>
      <td>13</td>
      <td>93</td>
      <td>34</td>
      <td>101</td>
      <td>25</td>
      <td>110</td>
      <td>37</td>
      <td>103</td>
      <td>28</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Mixed team</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>1</td>
      <td>41</td>
      <td>1</td>
      <td>58</td>
      <td>2</td>
      <td>50</td>
      <td>2</td>
      <td>46</td>
      <td>3</td>
      <td>35</td>
      <td>3</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>9</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>11</td>
      <td>5</td>
      <td>14</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>9</td>
      <td>3</td>
      <td>17</td>
      <td>3</td>
      <td>17</td>
      <td>7</td>
      <td>23</td>
      <td>3</td>
      <td>16</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kosovo</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Jordan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 12 columns</p>
</div>



Calculate the total number of medals won for each year and visualize them. (1pt)


```python
import numpy as np
for i in range(len(df2.columns)):
    plt.bar(df2.columns[i], np.sum(df2[df2.columns[i]]))
```


    
![png](outputs/output_16_0.png)
    


Are the total number medals won constant across years? Why or why not? (1pt)

Even if we consider the difference of winter and summer olympics, the number of medals won each year doesn't seem constant. It seems to be increasing each year. This may be because they are adding games to the olympics every year. 

To make a fair comparison of country performance, it makes more sense for us to look at the fraction of medals won per year. Convert your raw medal counts to the relative percentage of medals won per year and visualize it with another spaghetti plot (2pts). 


```python
df2.loc['Total'] = [np.sum(df2[df2.columns[i]]) for i in range(len(df2.columns))]
```


```python
df2
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994</th>
      <th>1996</th>
      <th>1998</th>
      <th>2000</th>
      <th>2002</th>
      <th>2004</th>
      <th>2006</th>
      <th>2008</th>
      <th>2010</th>
      <th>2012</th>
      <th>2014</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>13</td>
      <td>101</td>
      <td>13</td>
      <td>93</td>
      <td>34</td>
      <td>101</td>
      <td>25</td>
      <td>110</td>
      <td>37</td>
      <td>103</td>
      <td>28</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Mixed team</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>1</td>
      <td>41</td>
      <td>1</td>
      <td>58</td>
      <td>2</td>
      <td>50</td>
      <td>2</td>
      <td>46</td>
      <td>3</td>
      <td>35</td>
      <td>3</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>9</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>11</td>
      <td>5</td>
      <td>14</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>9</td>
      <td>3</td>
      <td>17</td>
      <td>3</td>
      <td>17</td>
      <td>7</td>
      <td>23</td>
      <td>3</td>
      <td>16</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>Gabon</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kosovo</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Jordan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>183</td>
      <td>842</td>
      <td>205</td>
      <td>927</td>
      <td>234</td>
      <td>927</td>
      <td>252</td>
      <td>958</td>
      <td>258</td>
      <td>962</td>
      <td>295</td>
      <td>973</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 12 columns</p>
</div>




```python
import matplotlib.pyplot as plt
plt.figure(figsize=(19,8))
for i in range(len(df2)-1):
    plt.plot(df2.iloc[i] / df2.iloc[-1])
```


    
![png](outputs/output_22_0.png)
    


### 3.  PCA, Eigenanalysis, and Identifying Archetypes (40 pts)

Let's fit a principal components analysis on the dataset, taking each country as the observational unit. 

Create a scree plot from the PCA results. (2pts)


```python
df_prop = df2.iloc[:-1]/df2.iloc[-1]
df_prop
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994</th>
      <th>1996</th>
      <th>1998</th>
      <th>2000</th>
      <th>2002</th>
      <th>2004</th>
      <th>2006</th>
      <th>2008</th>
      <th>2010</th>
      <th>2012</th>
      <th>2014</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>0.071038</td>
      <td>0.119952</td>
      <td>0.063415</td>
      <td>0.100324</td>
      <td>0.145299</td>
      <td>0.108954</td>
      <td>0.099206</td>
      <td>0.114823</td>
      <td>0.143411</td>
      <td>0.107069</td>
      <td>0.094915</td>
      <td>0.124358</td>
    </tr>
    <tr>
      <th>Mixed team</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>0.005464</td>
      <td>0.048694</td>
      <td>0.004878</td>
      <td>0.062567</td>
      <td>0.008547</td>
      <td>0.053937</td>
      <td>0.007937</td>
      <td>0.048017</td>
      <td>0.011628</td>
      <td>0.036383</td>
      <td>0.010169</td>
      <td>0.029805</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>0.049180</td>
      <td>0.008314</td>
      <td>0.034146</td>
      <td>0.009709</td>
      <td>0.047009</td>
      <td>0.005394</td>
      <td>0.055556</td>
      <td>0.007307</td>
      <td>0.034884</td>
      <td>0.004158</td>
      <td>0.037288</td>
      <td>0.007194</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>0.049180</td>
      <td>0.003563</td>
      <td>0.082927</td>
      <td>0.003236</td>
      <td>0.072650</td>
      <td>0.007551</td>
      <td>0.091270</td>
      <td>0.003132</td>
      <td>0.062016</td>
      <td>0.000000</td>
      <td>0.057627</td>
      <td>0.001028</td>
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
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001040</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001040</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Kosovo</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001028</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001028</td>
    </tr>
    <tr>
      <th>Jordan</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001028</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 12 columns</p>
</div>




```python
df_center = df_prop.apply(lambda x: x-x.mean())
df_center
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994</th>
      <th>1996</th>
      <th>1998</th>
      <th>2000</th>
      <th>2002</th>
      <th>2004</th>
      <th>2006</th>
      <th>2008</th>
      <th>2010</th>
      <th>2012</th>
      <th>2014</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>0.064327</td>
      <td>0.113241</td>
      <td>0.056703</td>
      <td>0.093612</td>
      <td>0.138588</td>
      <td>0.102242</td>
      <td>0.092495</td>
      <td>0.108111</td>
      <td>0.136699</td>
      <td>0.100357</td>
      <td>0.088204</td>
      <td>0.117646</td>
    </tr>
    <tr>
      <th>Mixed team</th>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>-0.001247</td>
      <td>0.041982</td>
      <td>-0.001833</td>
      <td>0.055856</td>
      <td>0.001836</td>
      <td>0.047226</td>
      <td>0.001225</td>
      <td>0.041305</td>
      <td>0.004916</td>
      <td>0.029671</td>
      <td>0.003458</td>
      <td>0.023093</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>0.042469</td>
      <td>0.001602</td>
      <td>0.027435</td>
      <td>0.002997</td>
      <td>0.040297</td>
      <td>-0.001318</td>
      <td>0.048844</td>
      <td>0.000595</td>
      <td>0.028172</td>
      <td>-0.002553</td>
      <td>0.030577</td>
      <td>0.000483</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>0.042469</td>
      <td>-0.003148</td>
      <td>0.076215</td>
      <td>-0.003475</td>
      <td>0.065938</td>
      <td>0.000840</td>
      <td>0.084558</td>
      <td>-0.003580</td>
      <td>0.055304</td>
      <td>-0.006711</td>
      <td>0.050916</td>
      <td>-0.005684</td>
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
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.005672</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.005672</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
    </tr>
    <tr>
      <th>Kosovo</th>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.005684</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.005684</td>
    </tr>
    <tr>
      <th>Jordan</th>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.006711</td>
      <td>-0.005684</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 12 columns</p>
</div>




```python
from sklearn.decomposition import PCA
from sklearn import preprocessing
pca = PCA() 
pca.fit(df_center.T) 
```




```python
per_var = 100*pca.explained_variance_ratio_
labels = ['PC' + str(x) for x in range(1, len(per_var))]

plt.figure(figsize=(10,8))
plt.plot(per_var)
plt.ylabel('Fraction of Variance Explained', fontsize=15)
plt.xlabel('Principal Component', fontsize=15)
plt.title('Scree Plot', fontsize=25)
plt.show()
```


    
![png](outputs/output_29_0.png)
    


Determine the number of principal components needed in order for the low rank approximation to recreate at least 80% of the variation present in the initial data. Call this number K. (2pts) 


```python
np.cumsum(per_var)
```




    array([ 70.02074876,  80.87482351,  87.72258774,  92.00740557,
            94.86767435,  96.88627858,  98.12298584,  98.97898431,
            99.47843759,  99.76051967, 100.        , 100.        ])



In order to recreate at least 80% of the variation present in the initial data, We will need k = 2.

For each of the following countries, calculate the rank K approximation:

-China

-Norway 

-Russia

-United States

(4pts total, 1 pt for each country).


```python
from numpy.linalg import svd
def rank_k_approx(image,k):
    """
    Performs SVD decomposition. Truncates singular values at kth index & returns rank k approximation of image.
    
    --------
    Outputs: reconst_matrix, array of singular values s
    """
    U,D,Vt = svd(image,full_matrices=False)
    matrix_k = np.dot(U[:,:k],np.dot(np.diag(D[:k]),Vt[:k,:]))
   
    return matrix_k, D, Vt

```


```python
mat = rank_k_approx(df_prop, 2)
mat = pd.DataFrame(mat[0], index=df_prop.index)
mat
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>9.859097e-02</td>
      <td>1.142202e-01</td>
      <td>9.280682e-02</td>
      <td>1.126480e-01</td>
      <td>1.094876e-01</td>
      <td>1.153917e-01</td>
      <td>9.980671e-02</td>
      <td>1.189831e-01</td>
      <td>1.090765e-01</td>
      <td>1.166769e-01</td>
      <td>1.000746e-01</td>
      <td>1.113635e-01</td>
    </tr>
    <tr>
      <th>Mixed team</th>
      <td>-8.005688e-18</td>
      <td>-3.393267e-18</td>
      <td>-7.761790e-18</td>
      <td>-3.048620e-18</td>
      <td>-7.867869e-18</td>
      <td>-2.910599e-18</td>
      <td>-7.330629e-18</td>
      <td>-2.661439e-18</td>
      <td>-7.123191e-18</td>
      <td>-2.589065e-18</td>
      <td>-6.478957e-18</td>
      <td>-2.541261e-18</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>3.519927e-03</td>
      <td>4.190048e-02</td>
      <td>1.861485e-03</td>
      <td>4.323967e-02</td>
      <td>1.048528e-02</td>
      <td>4.565790e-02</td>
      <td>8.539291e-03</td>
      <td>4.926381e-02</td>
      <td>1.504472e-02</td>
      <td>4.844263e-02</td>
      <td>1.416563e-02</td>
      <td>4.578575e-02</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>4.742079e-02</td>
      <td>1.135392e-02</td>
      <td>4.631182e-02</td>
      <td>8.989804e-03</td>
      <td>4.508379e-02</td>
      <td>7.635738e-03</td>
      <td>4.227156e-02</td>
      <td>5.355715e-03</td>
      <td>3.961511e-02</td>
      <td>5.097863e-03</td>
      <td>3.592801e-02</td>
      <td>5.385186e-03</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>7.710978e-02</td>
      <td>1.109302e-02</td>
      <td>7.558939e-02</td>
      <td>6.976924e-03</td>
      <td>7.202831e-02</td>
      <td>4.323033e-03</td>
      <td>6.776723e-02</td>
      <td>-6.204717e-05</td>
      <td>6.224460e-02</td>
      <td>-3.373767e-04</td>
      <td>5.635778e-02</td>
      <td>6.105340e-04</td>
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
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>-3.499201e-05</td>
      <td>1.617128e-04</td>
      <td>-4.070315e-05</td>
      <td>1.697322e-04</td>
      <td>-3.693318e-06</td>
      <td>1.811658e-04</td>
      <td>-8.815093e-06</td>
      <td>1.984877e-04</td>
      <td>2.091221e-05</td>
      <td>1.953553e-04</td>
      <td>2.112489e-05</td>
      <td>1.840482e-04</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>-3.499201e-05</td>
      <td>1.617128e-04</td>
      <td>-4.070315e-05</td>
      <td>1.697322e-04</td>
      <td>-3.693318e-06</td>
      <td>1.811658e-04</td>
      <td>-8.815093e-06</td>
      <td>1.984877e-04</td>
      <td>2.091221e-05</td>
      <td>1.953553e-04</td>
      <td>2.112489e-05</td>
      <td>1.840482e-04</td>
    </tr>
    <tr>
      <th>Kosovo</th>
      <td>-3.003075e-05</td>
      <td>1.510089e-04</td>
      <td>-3.540141e-05</td>
      <td>1.583422e-04</td>
      <td>-1.044226e-06</td>
      <td>1.689047e-04</td>
      <td>-5.957053e-06</td>
      <td>1.848946e-04</td>
      <td>2.155100e-05</td>
      <td>1.819675e-04</td>
      <td>2.155329e-05</td>
      <td>1.714662e-04</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>-3.003075e-05</td>
      <td>1.510089e-04</td>
      <td>-3.540141e-05</td>
      <td>1.583422e-04</td>
      <td>-1.044226e-06</td>
      <td>1.689047e-04</td>
      <td>-5.957053e-06</td>
      <td>1.848946e-04</td>
      <td>2.155100e-05</td>
      <td>1.819675e-04</td>
      <td>2.155329e-05</td>
      <td>1.714662e-04</td>
    </tr>
    <tr>
      <th>Jordan</th>
      <td>-3.003075e-05</td>
      <td>1.510089e-04</td>
      <td>-3.540141e-05</td>
      <td>1.583422e-04</td>
      <td>-1.044226e-06</td>
      <td>1.689047e-04</td>
      <td>-5.957053e-06</td>
      <td>1.848946e-04</td>
      <td>2.155100e-05</td>
      <td>1.819675e-04</td>
      <td>2.155329e-05</td>
      <td>1.714662e-04</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 12 columns</p>
</div>




```python
China = mat.loc['China']
Norway = mat.loc['Norway']
Russia = mat.loc['Russia']
USA = mat.loc['United States']
China, Norway, Russia, USA
```




    (0     0.027425
     1     0.072442
     2     0.024255
     3     0.073505
     4     0.037527
     5     0.076763
     6     0.033113
     7     0.081501
     8     0.042331
     9     0.080065
     10    0.039228
     11    0.075934
     Name: China, dtype: float64,
     0     0.117285
     1     0.017733
     2     0.114940
     3     0.011504
     4     0.109706
     5     0.007520
     6     0.103188
     7     0.000929
     8     0.094928
     9     0.000494
     10    0.085962
     11    0.001879
     Name: Norway, dtype: float64,
     0     0.088413
     1     0.081245
     2     0.084039
     3     0.079053
     4     0.094502
     5     0.080214
     6     0.086716
     7     0.081487
     8     0.091571
     9     0.079833
     10    0.083811
     11    0.076450
     Name: Russia, dtype: float64,
     0     0.098591
     1     0.114220
     2     0.092807
     3     0.112648
     4     0.109488
     5     0.115392
     6     0.099807
     7     0.118983
     8     0.109076
     9     0.116677
     10    0.100075
     11    0.111363
     Name: United States, dtype: float64)



For those 4 countries, plot the original data vs. the rank K approximation. (4 pts total, 1 pt per country)


```python
plt.figure()
plt.plot(df_prop.loc['China'], label = 'Original Data')
plt.plot(China, label = 'Rank K Approximation')
plt.title('China', size = 14)
plt.legend()
plt.figure()
plt.plot(df_prop.loc['Norway'], label = 'Original Data')
plt.plot(Norway, label = 'Rank K Approximation')
plt.title('Norway', size = 14)
plt.legend()
plt.figure()
plt.plot(df_prop.loc['Russia'], label = 'Original Data')
plt.plot(Russia, label = 'Rank K Approximation')
plt.title('Russia', size = 14)
plt.legend()
plt.figure()
plt.plot(df_prop.loc['United States'], label = 'Original Data')
plt.plot(USA, label = 'Rank K Approximation')
plt.title('USA', size = 14)

plt.legend()
plt.show()
```


    
![png](outputs/output_38_0.png)
    



    
![png](outputs/output_38_1.png)
    



    
![png](outputs/output_38_2.png)
    



    
![png](outputs/output_38_3.png)
    


Visualize the first two principal component vectors. (2 pts, one per PC vector)


```python
pca = PCA(n_components=2)
eigenvecs = pca.fit_transform(df_center.T)
eigenvecs_df = pd.DataFrame(data = eigenvecs, 
                            columns = [r'$\hat{\phi}_1$', r'$\hat{\phi}_2$'],
                            index = np.arange(1, 13, 1))
eigenvecs_df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$\hat{\phi}_1$</th>
      <th>$\hat{\phi}_2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.116016</td>
      <td>0.084868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.071038</td>
      <td>-0.002696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.113275</td>
      <td>0.033785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.082383</td>
      <td>0.019736</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.100152</td>
      <td>-0.038240</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.090466</td>
      <td>0.017516</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.090243</td>
      <td>-0.028629</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.103402</td>
      <td>-0.002625</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.072983</td>
      <td>-0.067136</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.103753</td>
      <td>0.007091</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.057976</td>
      <td>-0.007832</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.099604</td>
      <td>-0.015838</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8,6))
plt.plot(eigenvecs_df, label = eigenvecs_df.columns)
plt.xticks(range(12), labels=list(df_prop.columns))
plt.legend()
plt.show()
```


    
![png](outputs/output_41_0.png)
    


Focus on the first principal component for now. How can we interpret this principal component vector? (2pt)

This first principal component vector shows us that the variance in the winter Olympics is higher than the variance in the winter Olympics. 

If a country has a high PC1 score, what does that tell us about that country's historical performance at the summer and winter Olympics? (2pts)

If a country has a high PC1 score, that means their variance matches the direction of most important variance which in this case is positive during the winter Olympics. Therefore, having a high PC1 score would mean that countries variance of medals won is high during the winter and low during the summer. 

Compare the PC1 scores for China, Norway, Russia, and the United States. Which of them has the highest PC1 score? The lowest? (2pts) [Double check that this result aligns with your interpretation of the first principal component vector. ]


```python
scores = pd.DataFrame(data = np.dot(df_center, eigenvecs),
                      columns = ['PC1_score', 'PC2_score'],  
                      index =  df_center.T.columns)
```


```python
scores.loc[['China', 'Norway','Russia','United States']].sort_values('PC1_score')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1_score</th>
      <th>PC2_score</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>-0.024405</td>
      <td>-0.001448</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>-0.007112</td>
      <td>-0.008543</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>0.004235</td>
      <td>0.007116</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>0.055540</td>
      <td>0.003502</td>
    </tr>
  </tbody>
</table>
</div>



The country with the highest PC1 score is Norway and the country with the lowest PC2 score is the China. This aligns with my interpretation because China, Russia, and the United States seem like the typical powerhouse countries that are consistent in every Olympics but Norway seems like the type to be inconsistent with their medal winnings in the winter since sometimes they place and sometimes they don't, and their variance in the summer could be close to zero since they may barely place. 

#### The second principal component is harder to interpret, but there is a general trend in which earlier components are more positive and later components are closer to 0 or more negative. For this reason, we will interpret this component loosely as a general contrast between earlier years and more recent years. 

If a country has a high PC2 score, what does that tell us about that country's historical performance at the Olympics? (2pts)

If a country has a high PC2 score, that means their variance matches the direction of the second most important cause of variation which in this case we are saying is declining from previous years against more recent years. Therefore, if a country has a high PC2 score, its variance similarly begins to decline as the years become more recent. 

Create a scatter plot of the principal component scores, with the first PC score on the x-axis and the second PC score on the y-axis. (2pts)

Label the points with the name of their corresponding countries if the first PC score has an absolute value greater than 0.01 or the second PC score has a magnitude greater than 0.0025. (2pts)


```python
plt.figure(figsize=(13,7))
x = scores['PC1_score']
y = scores['PC2_score']
for i in range(len(x)):
    if((abs(x[i])>0.01) or (abs(y[i])>0.0025)):
        plt.scatter(x[i], y[i], label = scores.index[i])
    else:
        plt.scatter(x[i], y[i], color = 'black')
plt.legend()
plt.title(r'PC scores: '+r'$\xi_1$'+' vs. '+r'$\xi_2$', size = 15)
plt.xlabel('PC1_score', size = 14)
plt.ylabel('PC2_score', size = 14)
plt.show()
```


    
![png](outputs/output_53_0.png)
    


Let's take the 4 countries described above as "archetypal" countries. Describe the "profiles" for each of these archetypes, using the first two principal components to depict trends in historical performance at the Olympics. (8 pts total, 2 pts for each archetype.)

The first archetype like Norway would be countries whose variance slightly decreases over time and whose variance in higher in the winter than the summer. The second archetype like the United States would be countries whose variance slightly increases over time and whose variance is pretty consistent but slightly lower in the winter than the summer. The third archetype like China would be countries whose variance slightly increases over time whose variance is consistent in winter and summer but slightly increases in the summer slightly and decreases slightly in the winter. The last archetype would be like Russia and Italy, countries whose variance stays pretty consistent from winter to summer but decreases as the years go on. 
