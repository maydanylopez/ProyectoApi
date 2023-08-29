from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

#set FLASK_APP=main.py
#set FLASK_ENV=development

app = Flask(__name__)

#Carpeta definida para subir csv
DATA_FOLDER = 'data'
app.config['DATA_FOLDER'] = DATA_FOLDER
ALLOWED_EXTENSIONS = set(['csv'])

#Metedo para validar archivo
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#Creacion de la ruta de la API
@app.route('/getPredictionHeartDisease', methods=['POST'])
def getPredictionHeartDisease():
    if request.method == 'POST':
        #Valido que se envie la variable file
        if 'file' not in request.files:
            resp = jsonify({'message': 'No file part in the request', 'status': 400})
            resp.status_code = 400
            return resp
        
        #Leo la variable file del archivo csv
        files = request.files.getlist('file')
        errors = {}
        success = False

        # Subo archivo csv a la carpeta definida
        for file in files:
            # Valido que el archivo sea correcto
            if file and allowed_file(file.filename):
                # Subo archivo en carpeta designada y coloco un nombre 
                # https://www.kaggle.com/datasets/adepvenugopal/heart-disease-data
                file.save(os.path.join(app.config['DATA_FOLDER'], 'heart_data.csv'))
                success = True
            else:
                errors['error'] = 'File type is not allowed'

        if success:
            # Cargando data en pandas data frame
            heart_data = pd.read_csv("data/heart_data.csv")

            #Head -> Imprime dataframe del archivo csv 10 registros
            print(heart_data.head(10))
            # Shape -> Cuantas Filas y columnas tiene nuestro dataset
            print(heart_data.shape)
            # Describe -> Descripcion de nuestras columnas del dataset
            print(heart_data.describe())
            # Resumen de las columas y los tipos datos
            print(heart_data.info())
            # Verifico valores nulos
            print(heart_data.isnull().sum())
            # Verifico la distribución de la variable target
            print(heart_data['target'].value_counts())

            # Divido las demás columnas y la columna target
            X = heart_data.drop(columns = 'target', axis = 1)
            print(X.head())
            # 'X' contiene los datos de la tabla sin incluir la columna TARGET, que se usara despues para el aprendizaje
            Y = heart_data['target']
            print(Y.head())
            # 'Y' contiene sola la columna TARGET para validar resultado antes de realizar el modelo

            # ******* Divido los datos para entrenamiento y pruebas
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, stratify = Y, random_state = 3 )
            # 1. stratify distribuira 0 y 1 de manera uniforme, la prediccion será imparcial
            # 2. test_split indica una proporción sobre el tamaño de los datos de prueba en el conjunto de datos, 
            # lo que significa que el 30 por ciento de los datos son datos de prueba
            # 3. random_state informa sobre la aleatoriedad de los datos, y el número informa sobre su grado de aleatoriedad

            # Verifico la forma de los datos divididos
            print(X.shape, X_train.shape, X_test.shape)

            # ******* Modelo de aprendizaje    Regresión logística
            model = LogisticRegression()
            model.fit(X_train.values, Y_train)
            LogisticRegression()

            # Modelo de Evaluación
            # Precisión de los datos de entrenamiento
            # La función de precisión mide la precisión entre dos valores o columnas
            X_train_prediction = model.predict(X_train.values)
            training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
            print("La precisión de los datos de entrenamiento : ", training_data_accuracy)

            # Precisión de los datos de prueba
            X_test_prediction = model.predict(X_test.values)
            test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
            print("La precisión de los datos de prueba : ", test_data_accuracy)

            # Sistema de predicción de edificios
            # Valores de características de entrada

            # Validación parametro forme parte del request
            if 'age' not in request.form:
                resp = jsonify({'message': 'No age part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
        
            if 'gender' not in request.form:
                resp = jsonify({'message': 'No gender part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'chestPain' not in request.form:
                resp = jsonify({'message': 'No chestPain part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'trestbps' not in request.form:
                resp = jsonify({'message': 'No trestbps part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'chol' not in request.form:
                resp = jsonify({'message': 'No chol part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'fbs' not in request.form:
                resp = jsonify({'message': 'No fbs part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'restecg' not in request.form:
                resp = jsonify({'message': 'No restecg part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'thalach' not in request.form:
                resp = jsonify({'message': 'No thalach part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'exang' not in request.form:
                resp = jsonify({'message': 'No exang part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'oldpeak' not in request.form:
                resp = jsonify({'message': 'No oldpeak part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'slope' not in request.form:
                resp = jsonify({'message': 'No slope part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'ca' not in request.form:
                resp = jsonify({'message': 'No ca part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            if 'thal' not in request.form:
                resp = jsonify({'message': 'No thal part in the request.form', 'status': 400})
                resp.status_code = 400
                return resp
            
            # Validación de campos nulos
            if request.form['age'] == '':
                resp = jsonify({'message': 'Param Age is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                age = int(request.form['age'])

            if request.form['gender'] == '':
                resp = jsonify({'message': 'Param Gender is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                gender = int(request.form['gender'])
            
            if request.form['chestPain'] == '':
                resp = jsonify({'message': 'Param chestPain is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                chestPain = int(request.form['chestPain'])
            
            if request.form['trestbps'] == '':
                resp = jsonify({'message': 'Param trestbps is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                trestbps = int(request.form['trestbps'])
            
            if request.form['chol'] == '':
                resp = jsonify({'message': 'Param chol is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                chol = int(request.form['chol'])
            
            if request.form['fbs'] == '':
                resp = jsonify({'message': 'Param fbs is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                fbs = int(request.form['fbs'])
            
            if request.form['restecg'] == '':
                resp = jsonify({'message': 'Param restecg is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                restecg = int(request.form['restecg'])
            
            if request.form['thalach'] == '':
                resp = jsonify({'message': 'Param thalach is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                thalach = int(request.form['thalach'])
            
            if request.form['exang'] == '':
                resp = jsonify({'message': 'Param exang is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                exang = int(request.form['exang'])
            
            if request.form['oldpeak'] == '':
                resp = jsonify({'message': 'Param oldpeak is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                oldpeak = int(request.form['oldpeak'])
            
            if request.form['slope'] == '':
                resp = jsonify({'message': 'Param slope is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                slope = int(request.form['slope'])
            
            if request.form['ca'] == '':
                resp = jsonify({'message': 'Param ca is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                ca = int(request.form['ca'])
            
            if request.form['thal'] == '':
                resp = jsonify({'message': 'Param thal is required', 'status': 400})
                resp.status_code = 400
                return resp
            else:
                thal = int(request.form['thal'])

            # años, genero, cp, trestbps, colesterol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
            #  35     1      0     136       315       0     1        125     1       2        1     0    1

            input_data = (age, gender, chestPain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            #input_data = (60, 1, 0, 136, 315, 0, 1, 125, 1, 2, 1, 0, 1)
            print(input_data)
            # Cambio los datos de entrada en una matriz con numpy
            input_data_as_numpy_array = np.array(input_data)
            # Remodelo la matriz para predecir datos para una sola instancia
            reshaped_array = input_data_as_numpy_array.reshape(1,-1)

            # Predecir el resultado e imprimirlo
            prediction = model.predict(reshaped_array)
            print(prediction)
            
            # [0]: significa que el paciente tiene un corazón sano
            # [1]: significa que el paciente tiene un corazón enfermo

            if(prediction[0] == 0):
                result = "Paciente tiene un corazón saludable"
                print("Paciente tiene un corazón saludable 💛💛💛💛")
            else:
                result = "Paciente es propenso a tener problemas en el corazón"
                print("Paciente es propenso a tener problemas en el corazón 💔💔💔💔")

            #Genero la respuesta en JSON
            resp = jsonify({'message': 'CSV successfully upload', 'status': 200, 'result': result, 'input': input_data,})
            resp.status_code = 200
            return resp
        else:
            resp = jsonify({'message': errors, 'status': 500})
            resp.status_code = 500
            return resp

if __name__ == '__main__':
    app.run(debug=True, port=5000)