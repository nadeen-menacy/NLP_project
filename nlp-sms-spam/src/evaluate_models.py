import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocess import prepare_data, split_data, vectorize_tfidf

def evaluate_models():
    df = prepare_data("data/sms_spam.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vec = vectorize_tfidf(X_train, X_test)

    model_names = ["NaiveBayes", "LogisticRegression", "LinearSVC"]
    results = []

    for name in model_names:
        model = joblib.load(f"models/{name}.joblib")
        preds = model.predict(X_test_tfidf)
        report = classification_report(y_test, preds, output_dict=True)
        acc = report["accuracy"]
        f1 = report["weighted avg"]["f1-score"]
        results.append({"Model": name, "Accuracy": acc, "F1": f1})
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm).plot()
        plt.title(name)
        plt.show()

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.plot(x="Model", y=["Accuracy", "F1"], kind="bar")
    plt.title("Model Comparison")
    plt.ylim(0.8, 1.0)
    plt.show()

if __name__ == "__main__":
    evaluate_models()
