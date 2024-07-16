
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.HealthInsurancePrediction.logger_file.logger_obj import logger
from src.HealthInsurancePrediction.Exception.custom_exception import CustomException
from src.HealthInsurancePrediction.pipeline.prediction_pipeline import PredictionPipeline



# initializing the flask app

app = Flask(__name__)


# route to display the home page

@app.route('/predict', methods = ['POST', 'GET'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')
    

    else : 
        try:
            age = request.form.get('age')
            bmi = request.form.get('bmi')
            sex = request.form.get('sex')
            smoker = request.form.get('smoker')


            data = [age, bmi, sex, smoker]
            
            logger.info(f'-----------Feteched data successfully from the user--------------')
            

            data = np.array(data).reshape(1, 4)

            data = pd.DataFrame(data)

            print(data)

            obj = PredictionPipeline()

            results = obj.predictDatapoint(data)

            logger.info(f'-----------Below is the final result {results}------------------')

            print(results)

            return render_template('index.html', results = str(results))


        except Exception as e:
            raise CustomException(e, sys)
        



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True) ## http://127.0.0.1:5000