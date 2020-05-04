import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    
    Parameters:
    messages_filepath (str): The path to the messages csv file
    categories_filepath (str): The path to the categories csv file
    
    Returns:
    pandas.DataFrame: A dataframe from the merge of the two csv files
    """
    messages = pd.read_csv(messages_filepath).drop('original', axis=1)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    return df



def clean_data(df):
    """
    
    Parameters:
    df (pandas.DataFrame): The dataframe
    
    Returns:
    pandas.DataFrame: The cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    columns = [x.split('-')[0] for x in categories.values[0]]
    categories.columns = columns

    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df.drop('categories', axis=1), categories), axis=1)
    # load data from database
    df.drop_duplicates(inplace=True)
    df = df.loc[df.isnull().mean(1) < 0.5]

    return df


def save_data(df, database_filename):
    """
    Function that saves the df in a database. Note, if the table exist, it will be replaced.
    Parameters:
    df (pandas.DataFrame): dataframe to be saved in a database
    database_filename: The path to the database
    """
    database_filename = database_filename.lstrip('sqlite:///')
    engine = create_engine('sqlite:///%s' % database_filename)
    engine.execute("DROP TABLE IF EXISTS message_categories;")
    df.to_sql('message_categories', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()