from preprocess import prepare_data, split_data, vectorize_tfidf

df = prepare_data('data/sms_spam.csv')
print("✅ Data loaded:", df.shape)

X_train, X_test, y_train, y_test = split_data(df)
print("✅ Train/Test split:", len(X_train), "/", len(X_test))

X_train_tfidf, X_test_tfidf, vectorizer = vectorize_tfidf(X_train, X_test)
print("✅ TF-IDF shapes:", X_train_tfidf.shape, X_test_tfidf.shape)
