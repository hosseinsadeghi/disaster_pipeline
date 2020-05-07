import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, help="Location of the database with the cleaned training data",
                    default='sqlite:///../data/database.db')
parser.add_argument('--table', type=str, help="The name of the table in database", default='message_categories')
parser.add_argument('--model', type=str, help="Path to the best or selected classifier model", default='../models/best_model')

args = parser.parse_args()
_sql_database = args.database
table_name = args.table
model_name = args.model


def direct_mention():
    """
    This function goes through the data set of interest and looks at all the field in "message" column. 
    It then counts the number of times that the category name is mentioned directly in the test.
    
    Returns:
    pd.DataFrame: A dataframe with the number of times each category is mentioned directly in text 
    """
    engine = create_engine(_sql_database)
    df = pd.read_sql('SELECT * FROM "{}"'.format(table_name), engine)
    df = df.loc[df.isnull().mean(1) < 0.5]
    
    # select categories columns
    cols = df.columns[3:]
    datas = []

    for col in cols:
        datas.append(df['message'].apply(lambda x: int(sum([y in x for y in col.split('_')
                                                            if y not in ['and', 'other']]))))

    datas = pd.concat(datas, axis=1)
    datas.columns = [' '.join(col.split('_')) for col in cols]
    df = datas.sum().sort_values(ascending=False)
    df = df[df > df.std() / 2]
    return df


def categories_distribution():
    """
    This function load data from a SQL server and looks at the distribution of labels in data.
    The data is multi-lables, therefore sum(count) > len(data)
    Returns:
    pd.DataFrame: A dataframe with the number of appearance of each label
    """
    engine = create_engine(_sql_database)
    df = pd.read_sql('SELECT * FROM "{}"'.format(table_name), engine)
    df = df.loc[df.isnull().mean(1) < 0.5]
    cols = df.columns[3:]
    df = df[cols]
    df.columns = [' '.join(col.split('_')) for col in cols]
    df = df.sum().sort_values(ascending=False)
    df = df[df > df.std() / 2]
    return df


categories_distribution()
app = Flask(__name__)


def tokenize(text):
    """ 
    Function that tokenizes an input text string
    
    Parameters:
    text (str): Input text
    
    Returns:
    list (str): List of tokens extracted from the input text
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine(_sql_database)
df = pd.read_sql_table(table_name, engine)
mentions_categories = direct_mention()
model = joblib.load(model_name)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    y_axis = mentions_categories.values
    x_axis = list(mentions_categories.index)
    y_axis2 = categories_distribution()
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=x_axis,
                    y=y_axis
                )
            ],

            'layout': {
                'title': 'Number top category names appearing in messages',
                'yaxis': {
                    'title': "Number of category name appearance"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=x_axis,
                    y=y_axis2
                )
            ],

            'layout': {
                'title': 'Number of top categories appearing in data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.drop('child_alone', axis=1).columns[3:], classification_labels))
    classification_results.update({'child_alone': 0})
    keys = sorted(list(classification_results.keys()))
    left = [[k, classification_results[k]] for k in keys if classification_results[k]]
    right = [[k, classification_results[k]] for k in keys if not classification_results[k]]
    classification_results = left + right
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()