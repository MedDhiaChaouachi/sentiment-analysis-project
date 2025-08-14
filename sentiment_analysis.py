import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load Data
df = pd.read_csv("IMDB Dataset.csv")
print(df.head())

# 2. Preprocess
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['review'].apply(preprocess)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['sentiment'], test_size=0.2, random_state=42
)

# 4. Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
