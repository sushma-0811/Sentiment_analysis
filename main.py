# main.py
from src.data_preprocessing import preprocess_data
from src.model import train_model, evaluate_model

def main():
    raw_data_path = "data/raw_data2.csv"
    X_train, X_test, y_train, y_test = preprocess_data(raw_data_path)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Display results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
