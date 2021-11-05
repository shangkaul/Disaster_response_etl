import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This method loads the dataframes from csv files and merges them
    input: 
    messages_filepath - path to messages csv file
    categories_filepath - path to categories csv file
    return: 
    df - merged df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    This method cleans the data. Takes care of categorical columns and duplicate records.
    input: 
    df - df with data to be cleaned
    return: 
    df - cleaned df
    """
    #split categories
    categories = categories = df['categories'].str.split(';', expand = True)
    row = categories[0:1]
    category_colnames = row.apply(lambda x: x.str[:-2]).values[0]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    #review edit - remove non binary entries
    print("before{}".format(df.shape))
    df = df[df['related'] != 2]
    df=df.drop_duplicates()
    print("after removing non binary values {}".format(df.shape))
    
    return df

def save_data(df, database_filename):
    """
    This method saves the cleaned data in a sqlite db.
    input: 
    df - final cleaned df
    database_filename - db file path
    return: 

    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False,if_exists='replace')  

    return df


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