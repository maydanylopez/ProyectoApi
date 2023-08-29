# CVDPrediction
Proyecto Python para predicción de problemas en el corazón, modelo de entrenamiento con Machine Learning

# Pasos para levantar el API

En el terminal de Windows (cmd)
1. Crear el entorno virtual de Python

python -m venv CVDPrediction
cd CVDPrediction

2. Copiar archivos de Git en /CVDPrediction
3. Activar Ambiente
Scripts\activate

4. Instalar dependencias usando el requirements.txt
pip install -r requirements.txt

5. Establecer valores de variables globales para el proyecto
set FLASK_APP=main.py
set FLASK_ENV=development

6. Levantar el servidor
flask run
