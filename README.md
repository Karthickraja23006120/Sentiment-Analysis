# AI Echo: ChatGPT Sentiment Analysis 🤖💬

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-success.svg)

## 📌 Project Overview
This project performs sentiment analysis on user reviews of a ChatGPT application. The goal is to analyze customer feedback, understand sentiments, and classify reviews as **Positive**, **Neutral**, or **Negative**. It includes an end-to-end Machine Learning pipeline, from text preprocessing to model training and a fully interactive web dashboard for real-time predictions.

## 🛠️ Features
- **Data Preprocessing Pipeline:** Cleans text, removes stopwords via NLTK, and categorizes ratings into sentiments.
- **Exploratory Data Analysis (EDA):** Interactive visualizations uncovering insights about ratings, word frequencies, location-based trends, and platform comparisons.
- **Machine Learning Classification:** Utilizes TF-IDF vectorization and a Logistic Regression model to accurately classify reviews.
- **Web Dashboard:** A Streamlit application allowing users to view EDA and test the model on custom text inputs.

## 📊 Exploratory Data Analysis (EDA) Insights
The Streamlit dashboard answers several key analytical questions:
1. **Distribution of review ratings**: Shows the balance between satisfied and dissatisfied customers.
2. **Helpful Votes Distribution**: Highlights how helpful the reviews are to the community.
3. **Word Clouds**: Displays the most common words in positive vs. negative reviews, helping identify pain points and praised features.
4. **Average Rating Over Time**: Tracks user satisfaction over time.
5. **Ratings by Location**: Shows which regions have higher satisfaction.
6. **Platform Comparison**: Compares the average ratings between Web and Mobile users.
7. **Verified Purchase Satisfaction**: Checks if verified/paying users are more satisfied than free users.

## 🧠 Machine Learning Model
- **Feature Extraction**: Text data is converted to numerical representation using **TF-IDF Vectorization** (`TfidfVectorizer` from `scikit-learn` with a maximum of 1000 features).
- **Model**: A **Logistic Regression** model is trained on 80% of the dataset and validated on the remaining 20%.
- **Performance**: The model achieves high accuracy and generalizes well based on the keywords associated with positive and negative sentiments.
- **Artifacts**: The trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) are saved for quick inference.

## 🚀 Setup & Installation

### Prerequisites
Make sure you have Python installed. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn streamlit nltk matplotlib seaborn wordcloud
```

### Running the Project

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Train the Model**:
   Process the dataset and generate the machine learning artifacts (`model.pkl` and `vectorizer.pkl`).
   ```bash
   python train.py
   ```

3. **Launch the Dashboard**:
   Start the interactive Streamlit web application.
   ```bash
   python -m streamlit run app.py
   ```

## 📁 Repository Structure
```
├── app.py                      # Streamlit web dashboard application
├── train.py                    # Script to train the Machine Learning model
├── chatgpt_style_reviews_dataset.xlsx - Sheet1 (1).csv # Raw dataset
├── model.pkl                   # Trained Logistic Regression model (generated)
├── vectorizer.pkl              # Fitted TF-IDF Vectorizer (generated)
└── README.md                   # Project documentation
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
