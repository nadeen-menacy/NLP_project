from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train NB, LR, and SVM models and print evaluation metrics."""
    models = {
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC()
    }

    results = {}
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        results[name] = acc
        joblib.dump(model, f"models/{name}.joblib")

    return results
