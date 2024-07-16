

import pandas as pd
import joblib
import mlflow
import os
import mlflow.sklearn
from pathlib import Path
from src.HealthInsurancePrediction.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.HealthInsurancePrediction.utils.common import save_json
from src.HealthInsurancePrediction.logger_file.logger_obj import logger


class ModelEvaluation:
    def __init__(self, config : ModelEvaluationConfig):
        self.config = config


    
    def eval_metrics(self, actual, pred):
        r2score = r2_score(actual, pred)
        meanabsoluteerror = mean_absolute_error(actual, pred)
        meansquarederror = mean_squared_error(actual, pred)
        return r2score, meanabsoluteerror, meansquarederror
    


    def log_into_mlflow(self):

        logger.info(f'-----------Entered log_into_mlflow function----------------')

        test_data = pd.read_csv(self.config.test_data_path)

        model = joblib.load(self.config.model_path)

        logger.info(f'-----------successfully loaded model joblib--------------------------')

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        os.environ["MLFLOW_TRACKING_URI"]='https://dagshub.com/augustin7766/HealthInsurancePrediction_with_MLflow.mlflow'
        os.environ["MLFLOW_TRACKING_USERNAME"]="augustin7766"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="8a01ee4bec043666cf3ced22edc7d308526b4b42"

        mlflow.set_experiment('sxith_06_exp')

        with mlflow.start_run():

            logger.info(f'------------------mlflow function started--------------------------------')

            predicted = model.predict(test_x)

            (r2score, meanabsoluteerror, meansquarederror) = self.eval_metrics(test_y, predicted)

            scores = {'r2score' : r2score, 'meanabsoluteerror' : meanabsoluteerror, 'meansquarederror' : meansquarederror}

            save_json(path = Path(self.config.metric_file_name), data = scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('r2score', r2score)
            mlflow.log_metric('meanabsoluteerror', meanabsoluteerror)
            mlflow.log_metric('meansquarederror', meansquarederror)

            mlflow.sklearn.log_model(model, 'model', registered_model_name = 'RandomForestRegressor')

            logger.info(f'------------------------mlflow function completed-----------------------')

