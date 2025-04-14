from prometheus_client import start_http_server, Gauge, Counter, Summary
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import time

# Define Prometheus metrics
training_accuracy = Gauge('training_accuracy', 'Accuracy during training')
training_iterations = Counter('training_iterations', 'Number of training iterations')
model_training_time = Summary('model_training_time_seconds', 'Time taken to train the model')

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Function to train the model and track performance metrics
@model_training_time.time()
def train_model():
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model (Random Forest Classifier)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    training_accuracy.set(accuracy)  # Update Prometheus metric for accuracy

    # Increment the training iterations counter
    training_iterations.inc()

    # Print the results
    print(f"Training completed! Accuracy: {accuracy:.4f}")

# Start the Prometheus metrics server
if __name__ == '__main__':
    # Start the HTTP server on port 8000
    start_http_server(8000)
    print("Prometheus metrics server running on http://localhost:8000/metrics")

    # Train the model and expose metrics
    train_model()
