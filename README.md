## Weather Predictor for Travelers

### Overview

There are large amount of tourists traveling in the U.S. from all over the world. But for the people that's first time visiting the city or even first time come to America, they would have no idea about the weather, especially temperature, of the destination at the time they plan to travel. Moreover, some tourists might want to travel sometime and somewhere warm or cool, then it will be essential to know the average temperature of certain city and month to make travel plan. This tool is predicting the average temperatures and precipitation of the user's interest city and month. It will also display a histogram of 12 monthes average temperature in a year for the interested city to help the visitor to make travel plan. 

### Data description

Historical monthly mean temperature and precipitation data from 1999 to 2018 (20 years) for ten tourist cities in the U.S. + AMES has been obtained from [PRISM Climate](http://www.prism.oregonstate.edu/). **Seperate prediction models has been created for each location, the following documentation will take the analysis of Ames (Story, IA) data as an example.** The reason of choosing this year range is because we don't have full weather record of December 2019 yet. And we want to compare the prediction accuracy of most recent decade to late 90's decade, since climate change is noticed to be more dramatic in recent years. The data was collected from the explorer tab in the website with manually choosing needed location and data. Since the location is based on state and county, we don't have access to the original weather station data, it's hard to tell if the dataset for a county comes from more than one weather station or several neighbor counties share data from one weather station. This could be an accuracy concern for further predictions. The historical weather data is not hard to collect online for the selected year range, and it shouldn't change over time. The directly downloaded file contains data description at the top which cannot read in as dataframe of panda, so it has to be removed before load onto notebook. [Data details from PRISM.](http://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf)

### Explore the data

Before going into data analysis and model training, I imported necessary packages from numpy, pandas, matplotlib, seaborn, and sklearn. 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn as sk
import sklearn.datasets as skd
import sklearn.ensemble as ske
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
```

1. After loading the dataset, I fisrt take a look at the head(`df.head()`) and shape(`df.shape`) of the data. The original data have 5 columns includes Date (*year & month*), ppt(mm) (*average precipitation*), tmin(degrees C) (*minimum temperature*), tmean(degrees C) (*average temperature*) and tmax(degrees C) (*maximum temperature*); and 240 rows for 20 years and 12 monthes in each year. 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code1.png)

2. In order to feed year and month as independent variables into the model, the Date column need to be seperated into year and month columns. I used `df.str.split` function as showing below to seperate it. 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code2.png)

3. Check if there are any N/A in the data `pd.isnull()`. N/As are not found in the data. 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code3.png)

4. Finally, verify the quality of the data. Using `df.describe()` to get the numerical summary of all columns, and calculate & plot the correlation `corr` and distribution `attr` of each feature. Look for any zeros in the measurement columns and any anomalous data points in graphs. No zeros found in the data. The three temperatures have strong correlation with each other, so only mean temperature will be used for model prediction. 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code4.png)
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Plot1.png "Correlation Heatmap")
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Plot2.png "Distribution Scatter")
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Plot3.png "Distribution Hex")

### Model the data

To build a model for mean temperature and precipitation prediction, the overall procedures are define training and testing dataset, define and fit the model. In order to compare the predicting accuracy of most recent decade to late 90's decade, seperate models will be trained with 1999-2008 and 2009-2018 data. Besides, since there are two dependent variables (*mean precipitation & temperature*), each dependent variable would require a unique model. Thus, for AMES, four models are traininged for comparison. Following are detailed code. 

1. Average Temperature & Precipitation from 1999 to 2008 
```python
from sklearn.model_selection import train_test_split

# Select data from 1999 to 2008 and transpose column and row
dft = df[0:120].T

# Select year and month column for x
x = dft[5:7].T
# Select mean temperature for y1, mean precipitation for y2
y1 = dft[3:4].T
y2 = dft[1:2].T

# Print shape of x, y1 and y2
print (x.shape)
print (y1.shape)
print (y2.shape)
```
Out: 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code8.png)

```python
# Take a look at x, y1 and y2
x.head()
y1.head()
y2.head()
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code5.png)
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code6.png)
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code7.png)

```python
# Split training & testing set for temperature prediction
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, test_size = 0.3, random_state=1)
print(x1_train.shape)
print(y1_train.shape)
print(x1_test.shape)
print(y1_test.shape)

# Split training & testing set for precipitation prediction
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, test_size = 0.3, random_state=1)
print(x2_train.shape)
print(y2_train.shape)
print(x2_test.shape)
print(y2_test.shape)
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code9.png)
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code9.png)

```python
# Flat the arrays
y1_train = np.ravel(y1_train)
y2_train = np.ravel(y2_train)

# Fit temperature data
reg1 = ske.RandomForestRegressor(n_estimators= 1000, random_state= 0)
reg1.fit(x1_train, y1_train)

# Fit precipitation data
reg2 = ske.RandomForestRegressor(n_estimators= 1000, random_state= 0)
reg2.fit(x2_train, y2_train)
```
Out: 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code10.png)
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code10.png)

2. Average Temperature & Precipitation from 2009 to 2018
```python
# Define datasets for recent decade
dft = df[120:240].T
xr = dft[5:7].T
y1r = dft[3:4].T
y2r = dft[1:2].T

# Split training and testing dataset to train temperature model
x1r_train, x1r_test, y1r_train, y1r_test = train_test_split(xr, y1r, test_size = 0.3, random_state=1)
y1r_train = np.ravel(y1r_train)
reg1r = ske.RandomForestRegressor(n_estimators= 1000, random_state= 0)
reg1r.fit(x1r_train, y1r_train)

# Split training and testing dataset to train precipitation model
x2r_train, x2r_test, y2r_train, y2r_test = train_test_split(xr, y2r, test_size = 0.3, random_state=1)
y2r_train = np.ravel(y2r_train)
reg2r = ske.RandomForestRegressor(n_estimators= 1000, random_state= 0)
reg2r.fit(x2r_train, y2r_train)
```
Out: 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code10.png)
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code10.png)


### Communciate and visualize the results

1. Temperature Prediction from 1999 to 2008
```python
y1_pred = reg1.predict(x1_test)
y1_pred
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code11.png)

```python
print('Explained variance: ', explained_variance_score(y1_test, y1_pred))
print('Mean absolute error: ', mean_absolute_error(y1_test, y1_pred))
print('Mean squared error: ', mean_squared_error(y1_test, y1_pred))
print('R-square: ', r2_score(y1_test, y1_pred, multioutput = 'variance_weighted'))
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code12.png)

```python
tmean1 = df['tmean (degrees C)'].mean()
diff1 = y1_pred - tmean1
diff1
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code13.png)

```python
type(diff1)
sns.distplot(diff1, bins = 50)
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Plot9.png)

2. Precipitation Prediction from 1999 to 2008
```python
y2_pred = reg2.predict(x2_test)
print('Explained variance: ', explained_variance_score(y2_test, y2_pred))
print('Mean absolute error: ', mean_absolute_error(y2_test, y2_pred))
print('Mean squared error: ', mean_squared_error(y2_test, y2_pred))
print('R-square: ', r2_score(y2_test, y2_pred, multioutput = 'variance_weighted'))
tmean2 = df['ppt (mm)'].mean()
diff2 = y2_pred - tmean2
type(diff2)
sns.distplot(diff2, bins = 50)
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code14.png)

3. Temperature Prediction from 2009 to 2018
```python
y1r_pred = reg1r.predict(x1r_test)

print('Explained variance: ', explained_variance_score(y1r_test, y1r_pred))
print('Mean absolute error: ', mean_absolute_error(y1r_test, y1r_pred))
print('Mean squared error: ', mean_squared_error(y1r_test, y1r_pred))
print('R-square: ', r2_score(y1r_test, y1r_pred, multioutput = 'variance_weighted'))

tmean1r = df['tmean (degrees C)'].mean()
diff1r = y1r_pred - tmean1r
sns.distplot(diff1r, bins = 50)
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code16.png)

4. Precipitation Prediction from 2009 to 2018
```python
y2r_pred = reg2r.predict(x2r_test)

print('Explained variance: ', explained_variance_score(y2r_test, y2r_pred))
print('Mean absolute error: ', mean_absolute_error(y2r_test, y2r_pred))
print('Mean squared error: ', mean_squared_error(y2r_test, y2r_pred))
print('R-square: ', r2_score(y2r_test, y2r_pred, multioutput = 'variance_weighted'))

tmean2r = df['ppt (mm)'].mean()
diff2r = y2r_pred - tmean2r
sns.distplot(diff2r, bins = 50)
```
Out:
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code17.png)

5. Compute how much each feature contributes
```python
reg1.feature_importances_
```
Out: 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Code15.png)

```python
fet_ind = np.argsort(reg1.feature_importances_)
fet_imp = reg1.feature_importances_[np.argsort(reg1.feature_importances_)][::-1]
fig, ax = plt.subplots(1, 1, figsize = (6, 5))
labels = (['Month', 'Year'])
pd.Series(fet_imp, index= labels).plot('bar', ax=ax)
ax.set_title('Features importance')
```
Out: 
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Plot10.png)

The prediction accuracy of temperature in late 90's is calculated to be about 91.48%, and in the recent decade is about 95.57%. The plot of difference between predicted and average temperature data shows large difference, which indicate the model is necessary and useful for temperature prediction. The prediction accuracy of precipitation in late 90's is about 22.9%, and in the recent decade is about 21.8%. The plot of difference between predicted and average precipitation data shows little difference, which indicate the model is unnecessary and not useful for precipitation prediction. Besides, there is no clear trend and difference between the two decades. Part 5 shows that month has much higher weight in the trees than year, which is reasonable that temperature changes more in different month than year. Thus, base on the accuracy of each model, we can conclude that average temperature have much higher accuracy which is required and reliable. On the other hand, the accuracy of precipitation prediction is too low to show enough evidence for solid prediction. 

The analysis and prediction process are using weather data of AMES as an example, weather data of the other 10 most famouse tourist cities are available on github and model can be trained using the same code. The entire project is yet to be finished, since it's missing the user interaction part, where the program will take in interested city and month from the user and output predicted temperature and histogram on website. 

### Class Exercise

The homework has been created based on this project with clear comments and guide and it's uploaded on [github](https://github.com/joyleeisu/ABE516X-Project.git/ABE516X Project HW.ipynb). 

### WorkFlow
![alt text](https://github.com/joyleeisu/ABE516X-Project.git/Workflow.jpg)

