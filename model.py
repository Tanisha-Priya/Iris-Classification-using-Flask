import pandas as pd
from sklearn import datasets
import joblib 

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
import os
current = os.getcwd()
project = os.listdir()[0]
request_path = os.path.join(current, project)
os.chdir(request_path)
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
print("Accuracy: % {:10.2f}".format(accuracy * 100))

# Save Model
import joblib
joblib.dump(classifier, "saved_models/01.knn_with_iris_dataset.pkl")

# file_name_classifier ='01.knn_with_iris_dataset.sav'
# pickle.dump(classifier, open(file_name_classifier,'wb'))

# make predictions
# Read models
classifier_loaded = joblib.load("saved_models/01.knn_with_iris_dataset.pkl")
encoder_loaded = joblib.load("saved_models/02.iris_label_encoder.pkl")

# encoder_loaded = pickle.load(open(file_name,'rb'))
# classifier_loaded = pickle.load(open(file_name_classifier,'rb'))

# Prediction set
X_manual_test = [[4.0, 4.0, 4.0, 4.0]]
print("X_manual_test", X_manual_test)

prediction_raw = classifier_loaded.predict(X_manual_test)
print("prediction_raw", prediction_raw)

prediction_real = encoder_loaded.inverse_transform(classifier.predict(X_manual_test))
print("Real prediction", prediction_real)
