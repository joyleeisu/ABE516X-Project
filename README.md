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
![alt text]()

2. In order to feed year and month as independent variables into the model, the Date column need to be seperated into year and month columns. I used `df.str.split` function as showing below to seperate it. 
![alt text]()

3. Check if there are any N/A in the data `pd.isnull()`. N/As are not found in the data. 
![alt text]()

4. Finally, verify the quality of the data. Using `df.describe()` to get the numerical summary of all columns, and calculate & plot the correlation `corr` and distribution `attr` of each feature. Look for any zeros in the measurement columns and any anomalous data points in graphs. No zeros found in the data. The three temperatures have strong correlation with each other, so only mean temperature will be used for model prediction. 
![alt text]()
![alt text]()
![alt text]()
![alt text]()

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
(120, 2)
(120, 1)
(120, 1)

```python
# Take a look at x, y1 and y2
x.head()
y1.head()
y2.head()
```
Out:
![alt text]()
![alt text]()
![alt text]()

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
(84, 2)
(84, 1)
(36, 2)
(36, 1)

(84, 2)
(84, 1)
(36, 2)
(36, 1)

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
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
           
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

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
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
           
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)


### Communciate and visualize the results


What did you learn and do the results make sense?  Revisit your initial question and answer it. 

### Class Exercise

In each project, I'd like to see a homework assignment that the class can do/evaluate to learn more about your data.  This should be a reproducible notebook that allows them to learn one or more aspects of your data workflow.  It is also an opportunity to share your research with your colleagues.

Here is an example of a fantastic project website:

https://stephenslab.github.io/ipynb-website/

## Advanced Features

### Stylesheet (Advanced)

If you'd like to add your own custom styles:

1. Create a file called `/assets/css/style.scss` in your site
2. Add the following content to the top of the file, exactly as shown:
    ```scss
    ---
    ---

    @import "{{ site.theme }}";
    ```
3. Add any custom CSS (or Sass, including imports) you'd like immediately after the `@import` line

*Note: If you'd like to change the theme's Sass variables, you must set new values before the `@import` line in your stylesheet.*

### Layouts (Advanced)

If you'd like to change the theme's HTML layout:

1. [Copy the original template](https://github.com/pages-themes/slate/blob/master/_layouts/default.html) from the theme's repository<br />(*Pro-tip: click "raw" to make copying easier*)
2. Create a file called `/_layouts/default.html` in your site
3. Paste the default layout content copied in the first step
4. Customize the layout as you'd like

### Overriding GitHub-generated URLs (Advanced)

Templates often rely on URLs supplied by GitHub such as links to your repository or links to download your project. If you'd like to override one or more default URLs:

1. Look at [the template source](https://github.com/pages-themes/slate/blob/master/_layouts/default.html) to determine the name of the variable. It will be in the form of `{{ site.github.zip_url }}`.
2. Specify the URL that you'd like the template to use in your site's `_config.yml`. For example, if the variable was `site.github.url`, you'd add the following:
    ```yml
    github:
      zip_url: http://example.com/download.zip
      another_url: another value
    ```
3. When your site is built, Jekyll will use the URL you specified, rather than the default one provided by GitHub.

*Note: You must remove the `site.` prefix, and each variable name (after the `github.`) should be indent with two space below `github:`.*

For more information, see [the Jekyll variables documentation](https://jekyllrb.com/docs/variables/).


### Contributing (Advanced)

Interested in contributing to Slate? We'd love your help. Slate is an open source project, built one contribution at a time by users like you. See [the CONTRIBUTING file](docs/CONTRIBUTING.md) for instructions on how to contribute.

### Previewing the theme locally

If you'd like to preview the theme locally (for example, in the process of proposing a change):

1. Clone down the theme's repository (`git clone https://github.com/pages-themes/slate`)
2. `cd` into the theme's directory
3. Run `script/bootstrap` to install the necessary dependencies
4. Run `bundle exec jekyll serve` to start the preview server
5. Visit [`localhost:4000`](http://localhost:4000) in your browser to preview the theme

### Running tests

The theme contains a minimal test suite, to ensure a site with the theme would build successfully. To run the tests, simply run `script/cibuild`. You'll need to run `script/bootstrap` one before the test script will work.
