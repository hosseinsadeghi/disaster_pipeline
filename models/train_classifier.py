import sys
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer
import logging

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("logs"),
                        logging.StreamHandler()
])



def load_data(database_filepath):
    """
    Function that loads the data set from a SQL database
    
    Parameters:
    database_filepath (str): The path to the database
    
    Returns:
    tuple of dataframes: The training and test features, and training and test labels
    """
    if 'sqlite:///' in database_filepath:
        engine = create_engine(database_filepath)
    else:
        engine = create_engine('sqlite:///%s' % database_filepath)
        
    df = pd.read_sql('SELECT * FROM message_categories', engine)
    df = df.loc[df.isnull().mean(1) < 0.5]

    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'child_alone'], axis=1)
    Y = Y.applymap(lambda x: 1 if x >= 1 else 0)
    return train_test_split(X, Y, test_size=0.2)


def tokenize(text):
    """ 
    Function that tokenizes an input text string
    
    Parameters:
    text (str): Input text
    
    Returns:
    list (str): List of tokens extracted from the input text
    
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens


def build_model():
    """
    Function to build the full training, parameter search, and cross validation pipeline
    
    Returns:
    GridSearchCV instance with a full test extraction and random forest classifier
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize, use_idf=True)),
        ('sc', Normalizer()),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight='balanced', max_iter=10000)))
    ])

    params = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_df': [0.25, 0.5, 1.0],
    }
    return GridSearchCV(pipeline, param_grid=params, scoring='f1_macro', n_jobs=-1)



def evaluate_model(model, xtest, ytest, labels):
    """
    Given a model, test data, and labels, report the classification results for each label
    
    Parameters:
    model (BaseEstimator): A model with `predict` method that returns an array the same size as ytest
    xtest (pd.DataFrame): The test dataset
    ytest (pd.DataFrame): The test labels
    lables (list[str]): The list of labels for each class
    """
    ypred = model.predict(xtest)
    report_test = classification_report(ytest.values, ypred, target_names=labels.values)
    print(report_test)


def save_model(model, model_filepath):
    """
    Save model as pickle object
    
    Parameters:
    model (BaseEstimator): model to save
    model_filepath (str): Path to save to
    """
    with open(model_filepath, 'wb') as file_handle:
        pickle.dump(model, file_handle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        logging.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_train, X_test, Y_train, Y_test = load_data(database_filepath)
        category_names = Y_train.columns
        logging.info('Building model...')
        model = build_model()
        
        logging.info('Training model...')
        model.fit(X_train, Y_train)
        
        logging.info('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        logging.info('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        logging.info('Trained model saved!')

    else:
        logging.exception('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()