---

# ⭐ Sentiment Analysis on Amazon Reviews using Machine Learning

A machine learning project that classifies Amazon product reviews into **positive**, **negative**, or **neutral** sentiments using classical ML algorithms and feature extraction techniques.
This project identifies the best-performing model for sentiment classification by evaluating multiple algorithms with different text vectorization methods.

---

##  Abstract

In today’s digital era, customer feedback is a critical source of business intelligence. This project aims to **automate sentiment analysis** of Amazon product reviews using **machine learning techniques**.
After experimenting with multiple classifiers and vectorization methods, the **Nu-SVM** model achieved the **highest accuracy of 94%**, proving to be the most effective for sentiment prediction.
This analysis provides valuable insights into customer opinions, helping e-commerce platforms and sellers make **data-driven decisions**.

---

##  Features

*  Preprocessing and cleaning of raw Amazon review data
*  Feature extraction using **CountVectorizer** and **TF-IDF Vectorizer**
*  Model training with multiple ML algorithms:

  * Naive Bayes
  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
*  Hyperparameter tuning using **RandomizedSearchCV** and **Nu-SVM**
*  Evaluation using accuracy, precision, recall, and F1-score
*  Visualization with confusion matrices and sentiment distribution charts

---

##  Tech Stack

* **Programming Language:** Python
* **Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

---

## 📂 Dataset

The dataset was sourced from **[Kaggle](https://www.kaggle.com/)** under the search term *“Amazon Reviews.”*
Due to licensing restrictions, the dataset is **not included** in this repository.

After downloading, place it inside the `data/` folder as:

```
data/amazon_reviews.csv
```

---

##  Methodology

1. **Data Collection:** Downloaded Amazon product reviews dataset from Kaggle.
2. **Preprocessing:**

   * Handled missing values
   * Combined `review_title` and `review_text` into one column
   * Labeled sentiments based on ratings:

     * ⭐ 4–5 → Positive
     * ⭐ 3 → Neutral
     * ⭐ 1–2 → Negative
3. **Feature Extraction:** Applied **CountVectorizer** and **TF-IDF** to convert text into numerical form.
4. **Model Training:** Trained multiple classifiers — Naive Bayes, Logistic Regression, Random Forest, and SVM.
5. **Evaluation:** Compared models using accuracy, precision, recall, and F1-score.
6. **Fine-Tuning:** Optimized Random Forest using **RandomizedSearchCV** and SVM using **Nu-SVM**.
7. **Visualization:** Plotted sentiment distribution and confusion matrices for better understanding.

---

##  Results & Model Comparison

### 🔹 CountVectorizer Results

| Algorithm           | Accuracy | Best Precision  | Best Recall    | Best F1-Score   |
| ------------------- | -------- | --------------- | -------------- | --------------- |
| Naive Bayes         | 92%      | Positive (0.92) | Positive (1.0) | Positive (0.96) |
| Logistic Regression | 90%      | Neutral (1.0)   | Positive (1.0) | Positive (0.94) |
| **Random Forest**   | **93%**  | Neutral (1.0)   | Positive (1.0) | Positive (0.96) |
| **SVM**             | **93%**  | Negative (0.96) | Positive (1.0) | Positive (0.96) |

>  **Observation:** Both **Random Forest** and **SVM** achieved the **highest accuracy of 93%** using the **CountVectorizer**, outperforming Naive Bayes and Logistic Regression.

---

### 🔹 TF-IDF Vectorizer Results

| Algorithm           | Accuracy | Best Precision  | Best Recall     | Best F1-Score   |
| ------------------- | -------- | --------------- | --------------- | --------------- |
| Naive Bayes         | 91%      | Positive (0.93) | Positive (0.98) | Positive (0.95) |
| Logistic Regression | 92%      | Positive (0.93) | Positive (0.99) | Positive (0.96) |
| **Random Forest**   | **93%**  | Neutral (1.0)   | Positive (1.0)  | Positive (0.96) |
| **SVM**             | **93%**  | Positive (0.95) | Positive (0.98) | Positive (0.97) |

>  **Observation:** Similarly, both **Random Forest** and **SVM** achieved the **same top accuracy (93%)** using the **TF-IDF Vectorizer**, showing consistent performance across feature extraction methods.

---

### 🔹 Fine-Tuned Models

| Algorithm             | Accuracy | Precision (Positive) | Recall (Positive) | F1-Score (Positive) |
| --------------------- | -------- | -------------------- | ----------------- | ------------------- |
| Random Forest (Tuned) | 90%      | 0.90                 | 1.0               | 0.95                |
|  **🏆Nu-SVM (Best)**  | **94%**  | **0.94**             | **0.99**          | **0.97**            |

>  **Conclusion:** Both **Random Forest** and **SVM** performed equally well with 93% accuracy on both vectorizers, but after fine-tuning, **Nu-SVM** achieved the **highest accuracy of 94%**, making it the most effective model overall.

---

## 📚 Publication

This project was **officially published** at the
 *🎓3rd World Conference on Information Systems for Business Management (ISBM)* — **Springer LNNS Series (2024)**

 **Read the Published Paper:**
[🔗 Sentiment Analysis on Amazon Reviews Using Machine Learning Techniques – ResearchGate](https://www.researchgate.net/publication/392245615_Sentiment_Analysis_on_Amazon_Reviews_Using_Machine_Learning_Techniques)

---

##  Conclusion

* Both **Random Forest** and **SVM** achieved **93% accuracy** using both CountVectorizer and TF-IDF Vectorizer.
* Fine-tuning with **Nu-SVM** improved performance to **94%**, confirming it as the **best-performing model**.
* The project demonstrates how combining effective **vectorization** with **model optimization** can significantly boost sentiment classification accuracy.

---

##  Future Scope

*  Integrate **deep learning architectures** like **LSTM** and **BERT** for better contextual understanding.
*  Expand sentiment classification to **multiple languages**.
*  Implement **aspect-based sentiment analysis** to identify sentiments about specific product features.
*  Build a **real-time web application or dashboard** for live sentiment monitoring.

---

##  Contributors

 **👩‍💻 Indukuri Varsha
 👩‍💻 Neelakantamatam Shreya
👩‍💻 Gottumukkala Kavya**
 

---

##  Contact

 **LinkedIn: https://www.linkedin.com/in/varsha-indukuri-99904b254/** 

---

### ⭐ Don’t forget to star this repository if you found it helpful!

---
