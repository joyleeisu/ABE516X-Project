## Weather Predictor for Travelers

### Overview

There are large amount of tourists traveling in the U.S. from all over the world. But for the people that's first time visiting the city or even first time come to America, they would have no idea about the weather, especially temperature, of the destination at the time they plan to travel. Moreover, some tourists might want to travel sometime and somewhere warm or cool, then it will be essential to know the average temperature of certain city and month to make travel plan. This tool is predicting the average temperatures and precipitation of the user's interest city and month. It will also display a histogram of 12 monthes average temperature in a year for the interested city to help the visitor to make travel plan. 

### Data description

Historical monthly mean temperature and percipitation data from 1999 to 2018 (20 years) for ten tourist cities in the U.S. + AMES has been obtained from [PRISM Climate](http://www.prism.oregonstate.edu/). Seperate prediction models has been created for each location, the following documentation will take the analysis of Ames(Story, IA) data as an example. The reason of choosing the year range is because we don't have full weather record of december 2019 yet. And we want to compare the prediction accuracy of most recent decade to late 90's decade, since climate change is noticed more dramatic in recent years. The data was collected from the explorer tab in the website with manually choosing needed location and data. Since the location is based on state and county, we don't have access to the original weather station data, it's hard to tell if the dataset for a county comes from more than one weather station or several neighbor counties share data from one weather station. This could be an accuracy concern for further predictions. The historical weather data is not hard to collect online for the selected year range, and it shouldn't change over time. The directly downloaded file contains data description at the top which cannot read in as dataframe of panda, so it has to be removed before load onto notebook. 

### Explore the data


[Details](http://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf)
Demonstrate what you would do to describe the data and if it has any patterns or anomolies.  Make some plots.

### Model the data

Build a model, fit the model, validate the model.

### Communciate and visualize the results

What did you learn and do the results make sense?  Revisit your initial question and answer it.  H

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
