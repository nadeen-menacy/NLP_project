from preprocess import prepare_data, split_data, vectorize_tfidf
from models import train_and_evaluate
import joblib

# 1️⃣ Load and clean data
df = prepare_data('data/sms_spam.csv')

# 2️⃣ Split
X_train, X_test, y_train, y_test = split_data(df)

# 3️⃣ Vectorize with TF-IDF
X_train_tfidf, X_test_tfidf, vec = vectorize_tfidf(X_train, X_test)
joblib.dump(vec, "models/tfidf_vectorizer.joblib")

# 4️⃣ Train and evaluate
results = train_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test)
print("\n✅ Training complete! Summary:")
print(results)
