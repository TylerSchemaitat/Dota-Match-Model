import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import Dota_Data


def do_logistic():
    feature, label = Dota_Data.get_lists()
    print(feature.__len__())
    # Assuming your input data is X (2000, 10, 140) and labels are y (2000, 2)
    X = np.array(feature)
    y = np.array(label)

    # Reshape your input data into a 2D array
    X = X.reshape(5560, -1)
    y = np.argmax(y, axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    print(count_correct_and_incorrect_per_class(y_test, y_pred))
    print(compute_f1_score(y_test, y_pred))

def compute_f1_score(y_test, y_pred):
    return f1_score(y_test, y_pred)

def count_correct_and_incorrect_per_class(y_true, y_pred):
    classes = np.unique(y_true)
    counts = {f'correct_{c}': 0 for c in classes}
    counts.update({f'incorrect_{c}': 0 for c in classes})

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            counts[f'correct_{true}'] += 1
        else:
            counts[f'incorrect_{true}'] += 1

    return counts



if __name__ == "__main__":
    do_logistic()