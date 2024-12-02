# Import necessary modules and packages
from flask import Flask, request, jsonify, session, url_for, redirect, render_template
from flower_form import FlowerForm
import pandas as pd
import joblib 
from sklearn import datasets

import os
current = os.getcwd()
project = os.listdir()[0]
request_path = os.path.join(current, project)
os.chdir(request_path)

df = datasets.load_iris()

y  = df.target

df = pd.DataFrame(data = df.data,columns = df.feature_names)

# Feature matrix
X = df

# Label encoder
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# file_name ='02.iris_label_encoder.sav'
# pickle.dump(encoder, open(file_name,'wb'))

joblib.dump(encoder, "saved_models/02.iris_label_encoder.pkl")

# split test train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# train model
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Test model
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Save Model
import joblib
joblib.dump(classifier, "saved_models/01.knn_with_iris_dataset.pkl")

# file_name_classifier ='01.knn_with_iris_dataset.sav'
# pickle.dump(classifier, open(file_name_classifier,'wb'))

# make predictions
# Read models
classifier_loaded = joblib.load("saved_models/01.knn_with_iris_dataset.pkl")
encoder_loaded = joblib.load("saved_models/02.iris_label_encoder.pkl")


# The code loads the machine learning model (01.knn_with_iris_dataset.pkl) and label encoder (02.iris_label_encoder.pkl) using joblib.load.
# These models will be used for making predictions.

#classifier_loaded = joblib.load("saved_models/01.knn_with_iris_dataset.pkl")
#encoder_loaded = joblib.load("saved_models/02.iris_label_encoder.pkl")

# prediction function
# The make_prediction function takes the loaded model, encoder, and a JSON object containing the input features for a flower. 
def make_prediction(model, encoder, sample_json):
    # parse input from request
    SepalLengthCm = sample_json['SepalLengthCm']
    SepalWidthCm = sample_json['SepalWidthCm']
    PetalLengthCm = sample_json['PetalLengthCm']
    PetalWidthCm = sample_json['PetalWidthCm']

    # Make an input vector
    flower = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]

    # Predict
    prediction_raw = model.predict(flower)

    # Convert Species index to Species name
    prediction_real = encoder.inverse_transform(prediction_raw)

    return prediction_real[0]

# An instance of the Flask application is created using Flask(__name__).
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

# Route Definitions:
# The root route ("/") is defined using the @app.route decorator. This route handles both GET and POST requests.

@app.route("/", methods=['GET','POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['SepalLengthCm'] = form.SepalLengthCm.data
        session['SepalWidthCm'] = form.SepalWidthCm.data
        session['PetalLengthCm'] = form.PetalLengthCm.data
        session['PetalWidthCm'] = form.PetalWidthCm.data

        return redirect(url_for("prediction"))
    return render_template("home.html", form=form)


# Read models
# classifier_loaded = joblib.load("saved_models/01.knn_with_iris_dataset.pkl")
# encoder_loaded = joblib.load("saved_models/02.iris_label_encoder.pkl")

# The prediction route ("/prediction") is defined to display the prediction results
@app.route('/prediction')
def prediction():
    content = {'SepalLengthCm': float(session['SepalLengthCm']), 'SepalWidthCm': float(session['SepalWidthCm']),
               'PetalLengthCm': float(session['PetalLengthCm']), 'PetalWidthCm': float(session['PetalWidthCm'])}

    results = make_prediction(classifier_loaded, encoder_loaded, content)

    return render_template('prediction.html', results=results)

print("---------------PORT SUCCESS----------------")
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8883)
    app.run()
