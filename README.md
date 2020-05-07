<p align="center">
  <a href="https://www.google.com/search?q=what+is+distaster">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQTiEKjqxHURhDjy54nPKrElc8gYROGkq5LyMSRDgPZ7_yiR0Q0&usqp=CAU" alt="disaster logo" width="72" height="72">
  </a>
</p>

<h3 align="center">Distaster Response Pipline</h3>

<p align="center">
  A simple web application that identifies and labels stress signals
</p>


## Table of contents

- [Quick start](#quick-start)
- [Summary](#summary)
- [Contributing](#contributing)
- [Creators](#creators)
- [Thanks](#thanks)
- [Copyright and license](#copyright-and-license)


## Quick start
To get started, there are two options:
- Clone the repo: `git clone https://github.com/hosseinsadeghi/disaster_pipeline.git`
- Fork the repo and open in GitPod.

Read the section below.

### Instructions:
1. Install repos in requirements.txt. The code is tested for python >=3.6

`pip install -r requirements.txt`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        

2. Run the following command in the app's directory to run your web app.

    `python run.py`
    
To explore more option use `-h` or `--help`. For example if using a different database name, table name, and model name you can type:

`python run.py --database DATABASE --table TABLE --model MODEL`

3. Go to http://0.0.0.0:3001/

## Summary
This project has three main components:
- Data processing
- Machine learning pipeline (feature selection, model selection)
- Web application deployment

In the first part of the project, the data of messages and categories is processed, merged, cleaned and saved in a database. The categories of interest all appear in one line delimited by ';'. After exctracing the categories, we can see that there are 36 labels (non-exclusive) associated with each message.

This data is unbalanced, to deal with the minority class, one can under-sample the majority class. There are other methods to deal with unbalanced data as well. For example, SMOTE is a well established technique. You may also refer to [this](https://github.com/hosseinsadeghi/oversampling_vae) repo for an example of using VAEs to over-sample the minority class. A gentle introduction can be found [here](https://towardsdatascience.com/oversampling-with-vaes-e410887fe51) as well.

A few machine learning mdoels are picked and cross validated using GridSearch over a few choice of parameters. In the end, a SVM was used as a classifier. Because the data is multi-labeled, the classifier is wrapped in `MultiOutputClassifier`.

The saved model, the result of classificaiton, is used in the web application to make predictions. The web appliation accepts an inquery and displays the predicted labels. In the main page, two graphics show a few statistics about the data.

## Contributing

Please follow the common contribution rules!


## Creators

**Hossein Sadeghi**

- <https://github.com/hosseinsadeghi>
- <https://www.linkedin.com/in/hosseinsadeghi/>


## Thanks

<a href="https://www.udacity.com/">
  <img src="https://d20vrrgs8k4bvw.cloudfront.net/images/open-graph/udacity.png" alt="udacity" width="80" height="42">
</a>
<a href="https://github.com/hamidkhbl">
  <img src="https://avatars2.githubusercontent.com/u/35466381?s=460&u=fc6318e6bf181d8d14635476e2cce9d6315d9b63&v=4" alt="hamidgithub" width="42" height="42">
</a>

Thanks to [Hamid Khodabandehloo](https://github.com/hamidkhbl) for insightfull conversations, and the team at [Udacity](https://www.udacity.com/) for the great nanodegree course.


## Copyright and license

Code copyright 2020 the [Authors.](https://github.com/hosseinsadeghi/disaster_pipeline/graphs/contributors) Code released under the [Apache 2.0 License](https://github.com/hosseinsadeghi/disaster_pipeline/blob/master/LICENSE). 
