import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(sys.argv[1]) as shop:
        reader = csv.DictReader(shop)
        evidence = list(list())
        labels = list()
        for row in reader:
            evidence.append(make_list(row))
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_true = 0
    total_false = 0
    predicted_true = 0
    predicted_false = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total_true += 1
            if predictions[i] == 1:
                predicted_true += 1
        else:
            total_false += 1
            if predictions[i] == 0:
                predicted_false += 1
    return predicted_true / total_true, predicted_false / total_false


def make_list(row: dict):
    transformed = list()
    k = list(row.keys())
    transformed.append(int(row['Administrative']))
    transformed.append(float(row['Administrative_Duration']))
    transformed.append(int(row['Informational']))
    transformed.append(float(row['Informational_Duration']))
    transformed.append(int(row['ProductRelated']))
    transformed.append(float(row['ProductRelated_Duration']))  # productRelated_Duration
    transformed.append(float(row['BounceRates']))  # bounce rates
    transformed.append(float(row['ExitRates']))
    transformed.append(float(row['PageValues']))
    transformed.append(float(row['SpecialDay']))  # SpecialDay
    transformed.append(get_month(row['Month']))  # Month
    transformed.append(int(row['OperatingSystems']))
    transformed.append(int(row['Browser']))
    transformed.append(int(row['Region']))
    transformed.append(int(row['TrafficType']))  # Traffic Type
    transformed.append(1 if row['VisitorType'] == 'Returning_Visitor' else 0)
    transformed.append(1 if row['Weekend'] == 'TRUE' else 0)  # Weekend
    return transformed


def get_month(name):
    match name:
        case 'Jan':
            return 0
        case 'Feb':
            return 1
        case 'Mar':
            return 2
        case 'Apr':
            return 3
        case 'May':
            return 4
        case 'Jun':
            return 5
        case 'Jul':
            return 6
        case 'Aug':
            return 7
        case 'Sep':
            return 8
        case 'Oct':
            return 9
        case 'Nov':
            return 10
        case 'Dec':
            return 11


if __name__ == "__main__":
    main()
