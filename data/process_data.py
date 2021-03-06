import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the messages and categories dataframe and return the merged dataframe
    Uses pandas to read the messages and categories dataset
    Merge the two datasets using id column
    
    Arguments:
    messages_filepath -- Messages dataset file path
    categories_filepath -- Categories dataset file path
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df=pd.merge(messages,categories,on='id')
    return df

def clean_data(df):
     """Cleans the dataframe categories this performs transform step ETL
    Extracts a dataframe of the 36 individual category columns
    Selects the first row of the categories dataframe
    Set each value to be the last character of the string
    Converts column from string to numeric
    Concatenates the original dataframe with the new categories dataframe
    Removing duplicates
    
    Arguments:
    df -- Merged dataframe
    """
    # creating a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    
    # selecting the first row of the categories dataframe
    row = categories.iloc[0,:]  
    
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    categories.columns = category_colnames
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
        # converting column from string to numeric
        categories[column] = categories[column].astype(str)
        
    df=df.drop('categories',axis=1)
    
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # removing duplicates
    df=df.loc[~df.duplicated()]
    
    return df



def save_data(df, database_filename):
     """Saves the cleaned data in database
    
    Arguments:
    df -- cleaned dataframe
    database_filename -- data will be stored in this database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False)  


def main():
     """Main function which performs ETL on the datasets
    """
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