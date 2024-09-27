import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.downlaod('stopwords')
#nltk.download('wordnet')

def conversion(data):
    if (data['target'] == True and data['3_label_majority_answer'] == 'Agree') or (data['target'] == False and data['3_label_majority_answer'] == 'Disagree'):
        return 1
    else:
        return 0

def preprocess_tweet(tweet):

  # Lowercase the text
  tweet = tweet.lower()

  # Remove URLs and hashtags
  tweet = re.sub(r"http\S+|#\S+", "", tweet)

  # Remove mentions
  tweet = re.sub(r"@\S+", "@user", tweet)

  # Remove emojis (optional)
  tweet = re.sub(r"[^\w\s]", "", tweet)

  # Remove punctuation
  tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)

  # Remove extra spaces
  tweet = re.sub(r"\s+", " ", tweet).strip()

  # Remove stop words (optional)
  stop_words = set(stopwords.words("english"))
  tweet = ' '.join([word for word in tweet.split() if word not in stop_words])

  # Perform lemmatization
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(token) for token in tweet.split()]

  preprocessed_tweet = " ".join(tokens)

  return preprocessed_tweet



def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, header=0)
    df['label'] = df.apply(conversion, axis=1)
    df['clean'] = df['tweet'].apply(preprocess_tweet)

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    tokenized_tweets = df['clean'].astype(str).apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    max_length = max(len(tokens) for tokens in tokenized_tweets)

    X = pad_sequences(tokenized_tweets, maxlen=max_length, padding='post', truncating='post')
    labels = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, max_length, tokenizer
