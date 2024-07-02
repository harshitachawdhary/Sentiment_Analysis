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