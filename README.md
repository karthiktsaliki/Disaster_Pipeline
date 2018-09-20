# Disaster Response Pipeline Project


In this project I am dealing with the two datasets. The problem here is I have to create two pipelines for classifying the disaster responses First pipeline covers ETL(Extract, Transform, Load) and next pipeline for creating machine learing model on text data. This project follows satisfies PEP-8 Convention.


### Github Link

https://github.com/karthiktsaliki/Disaster_Pipeline

### Libraries used

* scipy and numpy: SciPy and Numpy are free and open-source Python library used for scientific computing and technical computing.

* pandas: Pandas is a software library written for the Python programming language for data manipulation and analysis.

* sklearn: Machine learning library for the Python programming language. It features for classification and regression.

* sqlalchemy: Used for saving data in sqlite database

* matplotlib: Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.


### Motivation

Generally, Creating pipeline automatically classifies the messages without human intervention. Let's dive in detail about the data in data understanding section.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. Machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Datasets

Categories: This data set is actually the classification of different messages in 36 categories
               * These are few categories -- related, request, offer, aid_related, medical_help, medical_products, search_and_rescue

messages: This data set consists the 26216 textual messages.
           * Genre column indicate which genre each message belong to

### Project Components
There are three components you'll need to complete for this project.

* ETL Pipeline: In a Python script, process_data.py, write a data cleaning pipeline that:
    * Loads the messages and categories datasets
    * Merges the two datasets
    * Cleans the data
    * Stores it in a SQLite database

* ML Pipeline: In a Python script, train_classifier.py, write a machine learning pipeline that:
    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file
    
* Flask Web App: To visualize training data and to present the results



### Files in repository

This repository contains

* app
   * template
      * master.html  # main page of web app
      * go.html  # classification result page of web app
   * run.py  # Flask file that runs app

* data
  * disaster_categories.csv  # data to process 
  * disaster_messages.csv  # data to process
  * process_data.py
  * InsertDatabaseName.db   # database to save clean data to

* models
  * train_classifier.py
  * classifier.pkl  # saved model 

* README.md


### Results

Accuracy, Precision and recall all were above 90 percent. After trying different algorithms including boosting finally I got pretty good results with linear SVC.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
