# Disaster Response Pipeline Project

## Project Description
In this project we aim to classify messages sent during disasters. We have 36 categories to classify th messages. This use case is crucial especially at the time of a disaster as in any emergency condition it is crucial to get help to people requesting aid. To do this it is essential to find these calls for help messages by eliminating the noise. We have built a model that classifies the messages so that help can be dispatched without wasting precious time. The model is trained on message data from actual disasters courtesy Figre Eight.

Finally, this project contains a web app where you can input a message and get classification results.

![Screenshot of Web App]![image](https://user-images.githubusercontent.com/21197883/140520922-a0b5f437-4c3b-4402-89da-85f41d2ee6d8.png)


## File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
                |-- requirements.txt
                
          |-- data
                |-- disaster_messages.csv
                |-- disaster_categories.csv
                |-- Disaster_ETL.db
                |-- ETL Pipeline Preparation.ipynb
          |-- models
                |-- MLclassifier.pkl
                |-- ML Pipeline Preparation.ipynb
          |-- README
~~~~~~~
## Installation
1. Clone the git repo 
2. Generate the db file and classifier using the jupyter notebooks for ETL and ML respectively.(optional)
3. You can run the app using the exising model/db files too.
4. Change directory to app.
5. run pip install -r requirements.txt to install all the required python packages.
6. run python run.py.
7. To serve the webapp go to http://127.0.0.1:3001/ (port 3001 may be different in some cases)
NOTE: Make sure you use 127.0.0.1 to serve the app and not 0.0.0.0 even if that is displayed on CLI.

## File Descriptions
1. App folder contains the flask template files and the run script.
2. Data folder contains the essential data files with the ETL preparation jupyter notebook.
3. Models folder contains ML classifier pkl file and the ML preparation jupyter notebook.
4. README file

## Executing Scripts:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `cd data`<br>
        `python process_data.py disaster_messages.csv disaster_categories.csv Disaster_ETL.db`
    - To run ML pipeline that trains classifier and saves
        `cd models`<br>
        `python train_classifier.py data/Disaster_ETL.db models/MLclassifier.pkl`

2. Run the following command in the app's directory to run your web app.(cd app then -)
    `python run.py`

3. Go to http://127.0.0.1:3001/
