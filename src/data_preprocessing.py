import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(raw_data_path, encoding='utf-8'):
    try:
        df = pd.read_csv(raw_data_path, encoding=encoding, header=None)
    except UnicodeDecodeError:
        print(f"Error: Unable to decode using {encoding} encoding.")
        raise

    # Assuming sentiment labels are in the first column and text data in the second
    X = df[5].astype(str)  # Convert to string if not already
    y = df[0].astype(str)  # Convert to string if not already

    # Ensure there are at least two unique classes
    if len(y.unique()) < 2:
        raise ValueError("There must be at least two unique classes for sentiment labels.")

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Encoding labels using LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train_vectorized, X_test_vectorized, y_train_encoded, y_test_encoded
