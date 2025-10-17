# optimized_models.py

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

# Load dataset
data = pd.read_csv('data/amazon_reviews.csv')
data.fillna('', inplace=True)
data['text'] = data['review_title'] + " " + data['review_text']

# Label sentiments
def label_sentiment(rating):
    if rating in [4, 5]:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

data['sentiment'] = data['review_rating'].apply(label_sentiment)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization (used for final tuning)
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ------------------------------
# 🔹 Random Forest Optimization
# ------------------------------
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

rf_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42)
rf_search.fit(X_train_tfidf, y_train)

rf_best = rf_search.best_estimator_
y_pred_rf = rf_best.predict(X_test_tfidf)
print("\n🌲 Random Forest (Tuned) Results")
print("Best Parameters:", rf_search.best_params_)
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ------------------------------
# 🔹 Nu-SVM (Best Model)
# ------------------------------
nu_svm = NuSVC(nu=0.5, kernel='rbf', gamma='scale')
nu_svm.fit(X_train_tfidf, y_train)
y_pred_nu = nu_svm.predict(X_test_tfidf)

print("\n🏆 Nu-SVM (Optimized SVM) Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_nu) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_nu))
