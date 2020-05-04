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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



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
    Y = df.drop(['id', 'message', 'genre'], axis=1)
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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    A class to handles transforamtion of text to a boolean that indicates the first word is a verb
    """
    def starting_verb(self, text):
        """
        Given a text, determine if the first word is a verb
        
        Parameters:
        text (str): Input string
        
        Returns:
        boolean: Wether the first word is a verb
        
        """
       
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0:
                continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    A class that handles transformation of data to an integer the size of text
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Function that takes an input Panda.Series and returns a new dataframe with the length of the text entries
        
        Parameters:
        X (pandas.Series): Input series of string
        
        Returns:
        pandas.DataFrame: A dataframe with a single column holding the length of each entry in X
        """
        return pd.DataFrame(X.apply(lambda x: len(x)).values)


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('vect', TfidfVectorizer(tokenizer=tokenize)),
            ('verb', StartingVerbExtractor()),
            ('length_extractor', TextLengthExtractor()),
        ])),
        ('clf', RandomForestClassifier(class_weight='balanced', n_jobs=-1))
    ])

    params = {
        'features__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
#         'features__vect__max_df': [0.5, 0.75, 1.0],
#         'features__vect__use_idf': [True, False],
#         'features__vect__max_features': [100],
#         'features__vect__max_features': [None, 5000, 10000],
#         'clf__n_estimators': [50, 100, 200],
#         'clf__max_depth': [None, 10, 20],
#         'clf__min_samples_split': [2, 3, 4],
    }
    return GridSearchCV(pipeline, params)


def evaluate_model(model, xtest, ytest, labels):
    ypred = model.predict(xtest)

    for idx, label in enumerate(labels.values):
        report_test = classification_report(ytest.values[:, idx], ypred[:, idx])
        print(idx, label, '\n', report_test)



def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file_handle:
        pickle.dump(model, file_handle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_train, X_test, Y_train, Y_test = load_data(database_filepath)
        category_names = Y_train.columns
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()