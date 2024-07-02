#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from googleapiclient.discovery import build

# Set up YouTube Data API credentials
API_KEY = 'AIzaSyAMzRxXhFu9YjAjxwh7cGXCn2E6QLupFJI'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()
    
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'author': comment['authorDisplayName'],
            'text': comment['textDisplay'],
            'likeCount': comment['likeCount'],
            'publishedAt': comment['publishedAt']
        })
    
    return pd.DataFrame(comments)

# Example usage:
video_id = 'X0zdAG7gfgs'
comments_df = get_comments(video_id)
comments_df.to_csv('youtube_comments.csv', index=False)
print(comments_df.head())



# Load the data
comments_df = pd.read_csv('youtube_comments.csv')

# Data Cleaning
comments_df.drop_duplicates(subset='text', inplace=True)
comments_df.dropna(subset=['text'], inplace=True)

# Save the cleaned data
comments_df.to_csv('cleaned_youtube_comments.csv', index=False)




from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark Session
spark = SparkSession.builder.appName("YouTubeSentimentAnalysis").getOrCreate()

# Load data
comments_df = spark.read.csv('cleaned_youtube_comments.csv', header=True, inferSchema=True)

# Show schema
comments_df.printSchema()
comments_df.show(5)




# Example transformation: Filter comments with more than 0 likes
filtered_comments_df = comments_df.filter(col('likeCount') > 0)

# Show transformed data
filtered_comments_df.show(5)



import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Stop-word removal
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

comments_df = pd.read_csv('cleaned_youtube_comments.csv')

# Apply preprocessing
comments_df['processed_text'] = comments_df['text'].apply(preprocess_text)
print(comments_df)


from sklearn.feature_extraction.text import TfidfVectorizer

# Convert the text data to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(comments_df['processed_text'])

# Convert to DataFrame for further processing
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())



import os
current_dir= os.getcwd()



from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd




analyzer = SentimentIntensityAnalyzer()





def get_sentiment_score(text):
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']




comments_df['sentiment_score'] = comments_df['processed_text'].apply(get_sentiment_score)




def categorize_sentiment(score):
    if score > 0.05:  # Common threshold for positive sentiment
        return 'positive'
    elif score < -0.05:  # Common threshold for negative sentiment
        return 'negative'
    else:
        return 'neutral'

# Apply the corrected function to categorize sentiment
comments_df['sentiment'] = comments_df['sentiment_score'].apply(categorize_sentiment)

# Display the categorized sentiments
print(comments_df[['sentiment_score', 'sentiment']].head())




import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Using a raw string literal
python_path = r"C:\Users\91825\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"

os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

# Create Spark session
spark = SparkSession.builder \
    .appName("TFIDF Example") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "2g") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .master("local[*]") \
    .getOrCreate()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Tokenization and TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comments_df['processed_text'])

#saving the vectorization
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
joblib.dump(vectorizer, vectorizer_path)

X_train, X_test, y_train, y_test = train_test_split(X, comments_df['sentiment'], test_size=0.2, random_state=42)



import matplotlib.pyplot as plt




import seaborn as sns
from wordcloud import WordCloud




print (comments_df['processed_text'])




import matplotlib.pyplot as plt
import seaborn as sns

# Example Visualization: Word cloud of the most frequent terms
from wordcloud import WordCloud

all_text = ' '.join(comments_df['processed_text'])
wordcloud = WordCloud(width=800, height=400).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



print(comments_df.columns)




# Display a sample of the processed text to ensure it is correctly processed
print(comments_df['processed_text'].head())




X_existing = vectorizer.transform(comments_df['processed_text'])
print(X_existing.shape)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")



# Load the vectorizer from the file
vectorizer = joblib.load(vectorizer_path)

# Transform the existing comments using the loaded vectorizer
X_existing = vectorizer.transform(comments_df['processed_text'])

# Predict sentiments for the existing comments
all_predictions = clf.predict(X_existing)

# Add predictions to the DataFrame
comments_df['predicted_sentiment'] = all_predictions

# Count positive and negative comments
positive_count = comments_df[comments_df['sentiment'] == 'positive'].shape[0]
negative_count = comments_df[comments_df['sentiment'] == 'negative'].shape[0]

print(f"Number of positive comments: {positive_count}")
print(f"Number of negative comments: {negative_count}")




sample_comments = comments_df['processed_text'].sample(5)
sample_X = vectorizer.transform(sample_comments)
sample_predictions = clf.predict(sample_X)

print("Sample comments:")
print(sample_comments)
print("Sample predictions:")
print(sample_predictions)




import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of positive, negative, and neutral comments
sentiment_counts = comments_df['sentiment'].value_counts()

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(sentiment_counts)))
plt.title('Sentiment Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()




from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Example text for each sentiment category (replace with your actual data)
positive_text = ' '.join(comments_df[comments_df['sentiment'] == 'positive']['processed_text'])
negative_text = ' '.join(comments_df[comments_df['sentiment'] == 'negative']['processed_text'])
neutral_text = ' '.join(comments_df[comments_df['sentiment'] == 'neutral']['processed_text'])

# Generate word clouds
wordcloud_positive = WordCloud(width=800, height=400).generate(positive_text)
wordcloud_negative = WordCloud(width=800, height=400).generate(negative_text)
wordcloud_neutral = WordCloud(width=800, height=400).generate(neutral_text)



from PIL import Image
import numpy as np

# Load the YouTube symbol image
youtube_symbol_mask = np.array(Image.open('youtube_symbol.png'))

# Create word clouds with the custom mask
wordcloud_positive_masked = WordCloud(width=800, height=400, mask=youtube_symbol_mask).generate(positive_text)
wordcloud_negative_masked = WordCloud(width=800, height=400, mask=youtube_symbol_mask).generate(negative_text)
wordcloud_neutral_masked = WordCloud(width=800, height=400, mask=youtube_symbol_mask).generate(neutral_text)





import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import numpy as np


# Generate word cloud function
def generate_wordcloud(text, mask=None):
    wordcloud = WordCloud(width=800, height=400, mask=mask).generate(text)
    return wordcloud

# Generate word clouds for each sentiment category
positive_text = ' '.join(comments_df[comments_df['sentiment'] == 'positive']['processed_text'])
negative_text = ' '.join(comments_df[comments_df['sentiment'] == 'negative']['processed_text'])
neutral_text = ' '.join(comments_df[comments_df['sentiment'] == 'neutral']['processed_text'])

# Custom shape for word clouds
youtube_symbol_mask = np.array(Image.open('youtube_symbol.png'))

wordcloud_positive = generate_wordcloud(positive_text, mask=youtube_symbol_mask)
wordcloud_negative = generate_wordcloud(negative_text, mask=youtube_symbol_mask)
wordcloud_neutral = generate_wordcloud(neutral_text, mask=youtube_symbol_mask)

# Function to convert word cloud to base64 image
def get_wordcloud_base64(wordcloud):
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# Main Streamlit app
def main():
    st.title('Sentiment Analysis Dashboard')

    # Sentiment distribution bar chart
    st.subheader('Sentiment Distribution')
    sentiment_counts = comments_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Word clouds for each sentiment category
    st.subheader('Positive Sentiment')
    st.image(get_wordcloud_base64(wordcloud_positive))

    st.subheader('Negative Sentiment')
    st.image(get_wordcloud_base64(wordcloud_negative))

    st.subheader('Neutral Sentiment')
    st.image(get_wordcloud_base64(wordcloud_neutral))

    # Data table
    st.subheader('Data Table')
    st.write(comments_df)  # Display data table

# Run the app
if __name__ == '__main__':
    main()
