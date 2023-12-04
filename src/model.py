from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):
    # Initialize and train the Support Vector Machine (SVM) model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report
