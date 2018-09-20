import sys
import pandas as pd
import string
import pickle
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))


def load_data(database_filepath):
     """Load the messages and categories merged dataframe from database
    Uses pandas to read the table for reading dataset    
    Arguments:
    database_filepath -- Database file path
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse.db',engine)
    x=df.message
    y=pd.DataFrame(df.iloc[:,4:])
    return x,y,list(y.columns)

def tokenize(text):
    """Tokenize the given text.
    Uses NLTK word_tokenize function to split the given text into words.
    Uses WordNetLemmatizer of NLTK to convert the given list of words into their base form.
    Removes punctuations from text
    
    Arguments:
    text -- Each message in messages dataset
    """
    # Remove punctuations
    for key in string.punctuation:
        text=text.replace(key,' ')
      
    # NLTK Tokenization and Lemmatization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Pipeline for creating model
    Uses both TfidfVectorizer and MultiOutputClassifier for creating ML pipeline
    Uses Grid Search for parameter tuning
    Return CV object
    """
    
    # TfidfVectorizer and MultiOutputClassifier for creating ML pipeline
    pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=tokenize,stop_words=eng_stopwords)),
    ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])
    
    # Grid Search for parameter tuning
    parameters =  {
                    'clf__estimator__estimator__class_weight': ['balanced',None],
                    'clf__estimator__estimator__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'clf__estimator__estimator__max_iter':[100,500,1000]
                  }
    
    cv =  GridSearchCV(pipeline, parameters) 
        
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the created model on test data
    Uses classification_report to print the accuracy, precision, recall
    
    Arguments:
    model -- final model to evaluate test data
    X_test -- independent data 
    Y_test -- Variable which we have to predict
    category_names -- list of categories
    """
    test_preds=model.predict(X_test)
    for ind,cat in enumerate(category_names):
        print('For Category: ',cat,'\n classification report is \n')
        print(classification_report(Y_test[cat].values,test_preds[:,ind]))
        print('\n')
        print('Accuracy Score: ',accuracy_score(Y_test[cat].values,test_preds[:,ind]))
        print('\n')
        print('Recall Score: ',recall_score(Y_test[cat].values,test_preds[:,ind]))
        print('\n')
        print('Precision Score: ',precision_score(Y_test[cat].values,test_preds[:,ind]))
        print('\n')


def save_model(model, model_filepath):
    """save the best model
    
    Arguments:
    model -- final model 
    model_filepath -- path to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Main method which covers
     Loading data
     Building model
     Training model
     Evaluating model
     Saving model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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