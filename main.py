from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from functions import ClassifierSystem, preprocess_data, profits_metric, accuracy_metric
import pandas

pandas.set_option('display.width', None)  # for printing the whole width of the dataframe

## OPTIONS ##
classifiers = ["linear-regression", "logistic-regression", "decision-tree", "gaussian", "multinomial", "bernoulli", "categorical"] # you can add randomforest, but it is slow
fixed_treshold = None   # Decide automatically or choose a fixed treshold
feature_selection = "backward"  # Keep all features or do "backward" or "forward" feature selection
metric = "profits"  # Choose between "profit" or "accuracy"

## STEP 1: PREPROCESSING ##
# Do preprocessing on the data (removing inappropriate columns, one-hot encoding, etc.)
existing, _ = preprocess_data("existing-customers.xlsx")

## STEP 2: SPLIT INTO TRAINING AND TESTING ##
# Split the data into training and testing data
# X is the data the classifier will use to make predictions, y is the label the classifier will try to predict
X_train, X_test, y_train, y_test = train_test_split(existing.drop(["class"], axis=1), existing["class"], test_size=0.2)

## LOOP OVER GIVEN CLASSIFIERS TO FIND THE BEST ##
best_classifier : tuple = (0, None, None)
assert len(classifiers) > 0
for classifier in classifiers:
    print(f"Testing classifier: {classifier}")
    try:
        ## STEP 3: CREATE CLASSIFIER SYSTEM ##
        system = ClassifierSystem(classifier, X_train, X_test, y_train, y_test, fixed_treshold)

        ## STEP 4: FEATURE SELECTION ##
        features = system.feature_selection(feature_selection, metric, treshold=0.24, silent=True)

        ## STEP 5: TRAINING ##
        # Train the model on the selected features of the training data
        model = system.train(features)

        ## STEP 6: TESTING ##
        # Test the model on the selected features of the testing data
        score = system.test(features, metric, 0.24)

        if score > best_classifier[0]:
            best_classifier = (score, system, features)
    except Exception as e:
        continue

print(f"Best classifier: {best_classifier[1].type}")

## STEP 7: Improving the treshold ##
# Adapting the treshold of the best classifier to optimize it further
system : ClassifierSystem = best_classifier[1]
features = best_classifier[2]
if fixed_treshold is None:
    # Try different tresholds and find the best one
    best_treshold, estimate_predictor = system.test_find_treshold(features, metric, silent=True)
    print(f"The optimal treshold is: {best_treshold}")
else:
    best_treshold, estimate_predictor  = system.test_find_treshold(features, metric, lower= fixed_treshold, upper = fixed_treshold, silent=True)
    print(f"Using the given treshold: {fixed_treshold}")




## STEP 8: USING THE CLASSIFIER ON UNKNOWN DATA ##
# load and preprocess new data in the same way as the data the classifier was trained on
system.predict_file("potential-customers.xlsx", "promotion_list.txt", features, best_treshold, estimate_predictor)