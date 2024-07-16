




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.HealthInsurancePrediction.logger_file.logger_obj import logger
from src.HealthInsurancePrediction.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts//model_trainer//model.joblib'))
        self.preprocessorObj = joblib.load(Path('artifacts//data_transformation//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'age', 1 : 'bmi', 2 : 'sex', 3 : 'smoker'})
            
            print(data_df)

            transformed_numeric_cols = self.preprocessorObj.transform(data_df)

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_numeric_cols)

            prediction = self.model.predict(transformed_numeric_cols)

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(prediction)

            return prediction
        
        
        except Exception as e:
            raise CustomException(e, sys)