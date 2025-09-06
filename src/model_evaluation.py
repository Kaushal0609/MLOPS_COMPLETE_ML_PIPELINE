import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging

#Ensure log directory exists

log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok=True)

logger = logging.getLogger('Model_Evaluation')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dirs, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime) s - %(name) s - %(levelname) s - %(message) s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from the file"""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model loaded succesfully from :%s",file_path)
        return model
    except FileNotFoundError as e:
        logger.error("File not found :%s",e)
        raise
    except Exception as e :
        logger.error("Unexpected error while loading the model :%s",e)
        raise

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

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics"""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)

        metrics_dict = {
            'accuracy':accuracy,
            'precision': precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpcted error during metrics calculation :%s",e)
        raise

def save_metrics(metrics: dict, file_path:str)->None:
    """Save the evaluatu=ion metrics to JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved int JSON file at :%s", file_path)
    except Exception as e :
        logger.error('Error occured while saving the metrics: %s',e)
        raise

def main():
    try:
        clf = load_model('models/models.pkl')
        test_data = load_data('data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'report/metrics.json')
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s",e)
        print(f"ERROR:{e}")
if __name__ == '__main__':
    main()
    