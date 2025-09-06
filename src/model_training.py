import pandas as pd
import os
import logging
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

#ensure the log directory exists

log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dirs, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime) s - %(name) s - %(levelname) s - %(message) s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)-> pd.DataFrame:
    """Load the data from csv"""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data is succesfully loaded from: %s',file_path)
        return df
    
    except pd.errors.ParserError as e :
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except FileNotFoundError as e :
        logger.error("File not found :%s",e)
        raise
    except Exception as e :
        logger.error('Failed to load the data: %s',e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) ->RandomForestClassifier:
    """Train the Random Forest model with hyperparameter"""
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("the number of rows in X_train and y_train must be same.")
        logger.debug("Initializing Random forest model with parameters: %s",params)

        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.debug('Model Training started with %d samples',X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("Model training completed")

        return clf
    except ValueError as e:
        logger.error("value error during model training: %s",e)
        raise
    except Exception as e:
        logger.error('Unexpected error occur while model training: %s',e)
        raise

def save_model(model, file_path:str) -> None:
    """Save the trained model in a file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model save to : %s', file_path)
    except FileNotFoundError as e:
        logger.error("FIle path not found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving the model: %s",e)
        raise

def main():
    try:
        params = {'n_estimators': 1000,
                  'random_state': 2}
        train_data = load_data('data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/models.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model training prosess: %s',e)
        print(f'Error:{e}')

if __name__ == '__main__':
    main()