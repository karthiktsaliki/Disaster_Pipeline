import json
import plotly
import pandas as pd
import operator
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar,Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
        
    # Initialize tokenizer and lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #Tokenize and lemmatize the given text
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
   """Renders landing page along with some analysis on training dataset
    Extracts the data needed for visuals
    Uses plotly for creating visuals
    renders the template
    """
    
    # extracting data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    col_val={}
    y=df.iloc[:,4:]
    for col in y:
        val=0
        if '1' in y[col].value_counts():
            val=y[col].value_counts()[1]
        if '2' in y[col].value_counts():
            val+=y[col].value_counts()[2]
        col_val[col]=val
    col_val = sorted(col_val.items(), key=operator.itemgetter(1),reverse=True)

    col_counts=[val[1] for val in col_val]
    col_names=[val[0] for val in col_val]
    

    # creating visuals using plotly
    graphs = [
        {
            # This bar plot is for knowing the distribution of messages with Genre 
            
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # This bar plot is for knowing how messages spread across categories
         {
            'data': [
                Bar(
                    y=col_counts,
                    x=col_names,
                )
            ],

            'layout': {
                'title': 'Count of messages in each category',
                'yaxis': {
                    'title': "Count"
                },
            }
        },
        # This pie plot is for knowing how messages relative spread across categories
        {
            'data':[
                Pie(
                     labels=col_names,
                     values=col_counts,
                    textposition='inside'
                )
            ],
            'layout':{
                'title':' Percentage distribution of message in each category',
                'orientation':'h'
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
     """Renders results page along with classified labels
    Saves user input in query
    Uses model to predict classification for query
    renders the template
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Main function to start the server
    Default port is 3001
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()