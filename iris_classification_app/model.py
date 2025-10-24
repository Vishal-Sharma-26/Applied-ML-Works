from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle


def load_and_train_model():
    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model (for deployment)
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X, y, iris.target_names


def predict_species(model, input_data):
    return model.predict(input_data)[0]