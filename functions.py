from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import accuracy_score, confusion_matrix

TP_profits = (0.1 * (980 - 10)) + (0.9 * (-10))  # When a high class customer was invited
FP_loss = (0.05 * (-310 - 10)) + (0.95 * (-10))  # When a low class customer was invited


def profits_metric(TP, FP, TN, FN):
    max_profit = ((TP + FN) * TP_profits)  # When all high class customers were invited correctly
    real_profit = (TP * TP_profits) + (FP * FP_loss)
    return real_profit, max_profit


def accuracy_metric(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)


def preprocess_data(filename):
    # remove colums that are not appropriate for training a model
    # remove redundant columns
    # remove columns that are irrelevant to the model
    existing = pandas.read_excel(filename).drop(
        ["race", "sex", "age", "native-country", "relationship", "education"], axis=1).dropna()
    rowIDs = existing["RowID"]
    existing.drop(["RowID"], axis=1, inplace=True)

    # Replacing string columns with numerical columns (one-hot encoding)
    temp = pandas.get_dummies(existing[["workclass", "marital-status", "occupation"]], drop_first=True)
    existing.drop(["workclass", "marital-status", "occupation"], axis=1, inplace=True)
    existing = pandas.concat([existing, temp], axis=1)
    if "class" in existing.columns:
        existing["class"] = existing["class"].map({"<=50K": 0, ">50K": 1})
    return existing, rowIDs.tolist()


class ClassifierSystem:
    def __init__(self, t, X_train, X_test, y_train, y_test, fixed_treshold):
        self.type = t
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.fixed_treshold: float = fixed_treshold

        self.trained_model = None

    def train(self, features):
        X_train = self.X_train[features]
        model = self.create_model(self.X_test[features])
        model.fit(X_train, self.y_train)
        self.trained_model = model

    def test(self, features, metric, treshold: float, silent=False):
        if self.trained_model is None:
            raise "Model not trained"
        X_Test = self.X_test[features]
        try:
            predictions = self.trained_model.predict_proba(X_Test)
            predictions = (predictions[:, 1] > treshold).astype(int)
        except:
            predictions = self.trained_model.predict(X_Test)
            predictions = (predictions[:] > treshold).astype(int)

        if metric == "profits":
            cm = confusion_matrix(self.y_test.round(), predictions.round())
            TP, FP, TN, FN = cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]
            real, maximum = profits_metric(TP, FP, TN, FN)
            result = real / maximum
            if not silent:
                print(
                    f"\t{self.type} profits score: Real profits: €{real} / Maximum profits: €{maximum} ({result * 100:.2f}%)")
            return result
        elif metric == "accuracy":
            result = accuracy_score(self.y_test, predictions)
            if not silent:
                print(f"\t{self.type} accuracy score: {result * 100:.2f}")
            return result

    def test_find_treshold(self, features, metric, lower=0.17, upper=0.6, silent=False):
        if self.trained_model is None:
            raise "Model not trained"
        X_Test = self.X_test[features]
        try:
            predictions = self.trained_model.predict_proba(X_Test)
        except:
            predictions = self.trained_model.predict(X_Test)

        best_treshold = lower, 0, 0, 0, 0, 0
        treshold = lower
        while treshold <= upper:
            try:
                predictions_final = (predictions[:, 1] > treshold).astype(int)
            except:
                predictions_final = (predictions[:] > treshold).astype(int)
            result = 0

            cm = confusion_matrix(self.y_test.round(), predictions_final.round())
            TP, FP, TN, FN = cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]
            if metric == "profits":
                real, maximum = profits_metric(TP, FP, TN, FN)
                result = real / maximum
                if not silent:
                    print(
                        f"\t{self.type} profits score: Real profits: €{real} / Maximum profits: €{maximum} ({result * 100:.2f}%)")
            elif metric == "accuracy":
                result = (TP + TN) / (TP + FP + TN + FN)
                if not silent:
                    print(f"\t{self.type} accuracy score: {result * 100:.2f}")

            if result > best_treshold[1]:
                best_treshold = treshold, result, TP, FP, TN, FN
            treshold += 0.01

        # Now we can calculate the expected profits
        TP, FP, TN, FN = best_treshold[2], best_treshold[3], best_treshold[4], best_treshold[5]
        real, maximum = profits_metric(TP, FP, TN, FN)
        temp = real / (TP + FP)

        return best_treshold[0], temp

    def predict_file(self, f_in, f_out, features, treshold: float, estimate_ratio):
        if self.trained_model is None:
            raise "Model not trained"
        unknown, rowIDs = preprocess_data(f_in)
        unknown = unknown[features]

        # feed the new data to the classifier
        try:
            result = self.trained_model.predict_proba(unknown)
            result = (result[:, 1] > treshold).astype(int)
        except:
            result = self.trained_model.predict(unknown)
            result = (result[:] > treshold).astype(int)

        customers = []

        print("\n### USING THE CLASSIFIER ON UNKNOWN DATA ###")
        for x in range(len(result)):
            if result[x].round() == 1:
                customers.append(rowIDs[x])

        estimate = estimate_ratio * len(customers)
        with open(f_out, "w") as file:
            file.write(
                f"This list consists of the rowIDs of {len(customers)} potential customers. Estimated profits are €{estimate:.2f}.\n")
            for customer in customers:
                file.write(customer + "\n")

        print(f"The rowIDs of the potential customers are written to {f_out}.")
        print(f"The estimated profits for this prediction: €{estimate:.2f}.")

    def create_model(self, X_test=None):
        if self.type == "linear-regression":
            return LinearRegression()
        elif self.type == "logistic-regression":
            return LogisticRegression()
        elif self.type == "decision-tree":
            return DecisionTreeClassifier()
        elif self.type == "knn":
            return KNeighborsClassifier()
        elif self.type == "gaussian":
            return GaussianNB()
        elif self.type == "multinomial":
            return MultinomialNB()
        elif self.type == "bernoulli":
            return BernoulliNB()
        elif self.type == "complement":
            return ComplementNB()
        elif self.type == "categorical":
            if X_test is None:
                X_test = self.X_test
            return CategoricalNB(min_categories=X_test.nunique())
        elif self.type == "randomforest":
            return RandomForestClassifier()
        else:
            raise ("Invalid classifier type")

    def feature_selection(self, method, metric, treshold=None, silent=False):
        if method is None:
            return list(self.X_train.columns)
        elif method == "forward":
            return self.forward_feature_selection(metric)
        elif method == "backward":
            return self.backward_feature_selection(metric)
        else:
            raise "Invalid feature selection method"

    def performance_with_features(self, features, metric):
        X_train_subset = self.X_train[features]
        X_test_subset = self.X_test[features]
        model = self.create_model(X_test_subset)
        model.fit(X_train_subset, self.y_train)

        try:
            predictions = model.predict_proba(X_test_subset)
            predictions = (predictions[:, 1] > 0.25).astype(int)
        except:
            predictions = model.predict(X_test_subset)
            predictions = (predictions[:] > 0.25).astype(int)

        if metric == "profits":
            cm = confusion_matrix(self.y_test.round(), predictions.round())
            TP = cm[1, 1]
            FP = cm[0, 1]
            TN = cm[0, 0]
            FN = cm[1, 0]
            real_profit, max_profit = profits_metric(TP, FP, TN, FN)
            return model, real_profit / max_profit
        elif metric == "accuracy":
            return model, accuracy_score(self.y_test.round(), predictions.round())
        else:
            raise "Invalid metric"

    def forward_feature_selection(self, metric):
        # Returns a list of features selected by the forward feature selection algorithm

        # Training the model with all features
        remaining_features = list(self.X_train.columns)
        model_all, score_all = self.performance_with_features(remaining_features, metric)

        # Feature selection
        best = None, 0
        chosen_features = []
        while True:
            # Choose the best feature that can be added
            best_feature = None
            for feature in remaining_features:
                model, score = self.performance_with_features(chosen_features + [feature], metric)
                if score > best[1]:
                    best = model, score
                    best_feature = feature

            if best_feature is None and best[1] < score_all:
                return list(self.X_train.columns)
            elif best_feature is None:
                return chosen_features
            else:
                chosen_features.append(best_feature)
                remaining_features.remove(best_feature)

    def backward_feature_selection(self, metric):
        # Returns a list of features selected by the backward feature selection algorithm
        # Removes one feature at a time and checks if the accuracy improves

        # Training the model with all features
        remaining_features = list(self.X_train.columns)
        model_all, score_all = self.performance_with_features(remaining_features, metric)

        # Feature selection
        score_remaining_features = score_all
        while True:
            worst_feature = None
            for feature in remaining_features:
                model, score = self.performance_with_features([f for f in remaining_features if f != feature], metric)
                if score > score_remaining_features:
                    score_remaining_features = score
                    worst_feature = feature

            if worst_feature is None:
                return remaining_features
            else:
                remaining_features.remove(worst_feature)
