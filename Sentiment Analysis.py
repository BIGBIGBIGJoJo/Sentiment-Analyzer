import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

#Load data
DATA_PATH = "sentiment140.csv"  # ← put file here
df = pd.read_csv(DATA_PATH, encoding="latin-1", header=None, usecols=[0, 5])
df.columns = ["label", "text"]

df = df[df["label"].isin([0, 4])]
df["label"] = df["label"].map({0: 0, 4: 1})
print(f"Loaded {len(df):,} tweets")

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\S+|https\S+|@\w+|#", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    tokens = word_tokenize(t)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["cleaned"] = df["text"].apply(clean_text)

#Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Train
model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Negative", "Positive"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
