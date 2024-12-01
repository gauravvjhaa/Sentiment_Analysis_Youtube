import re
import emoji
import time
import pandas as pd
import numpy as np
from collections import Counter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from tqdm import tqdm
import streamlit as st

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# YouTube API key

api_key = 'AIzaSyD8Yt4bQT44sI1Q-_VaBHXZ_hp6mLQZ44A'
youtube = build('youtube', 'v3', developerKey=api_key)

# Function to extract video ID from URL
def extract_video_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|live\/|shorts\/|\S*?[?&]" \
            r"v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        st.error("Invalid YouTube URL")
        st.stop()

# Function to fetch comments and replies from a YouTube video
def get_comments(video_id):
    comments = []
    next_page_token = None

    st.info("Fetching comments and replies...")
    while True:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100
        )
        try:
            response = request.execute()
        except HttpError as e:
            st.error(f"An HTTP error {e.resp.status} occurred: {e.content}")
            return pd.DataFrame()

        for item in response['items']:
            # Top-level comment
            comment = item['snippet']['topLevelComment']['snippet']
            comment_id = item['snippet']['topLevelComment']['id']
            comments.append({
                'comment_id': comment_id,
                'text': comment['textOriginal'],
                'author': comment.get('authorDisplayName', ''),
                'published_at': comment.get('publishedAt', ''),
                'like_count': comment.get('likeCount', 0),
                'reply_to': None  # Top-level comment
            })

            # Replies to the top-level comment
            if 'replies' in item:
                for reply in item['replies']['comments']:
                    reply_snippet = reply['snippet']
                    reply_id = reply['id']
                    comments.append({
                        'comment_id': reply_id,
                        'text': reply_snippet['textOriginal'],
                        'author': reply_snippet.get('authorDisplayName', ''),
                        'published_at': reply_snippet.get('publishedAt', ''),
                        'like_count': reply_snippet.get('likeCount', 0),
                        'reply_to': comment_id  # Reply to top-level comment
                    })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    st.success(f"Total comments and replies fetched: {len(comments)}")
    return pd.DataFrame(comments)

# Function to perform sentiment analysis
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    st.info("Analyzing sentiments...")
    for text in tqdm(df['text']):
        # VADER Sentiment
        vs = analyzer.polarity_scores(text)
        # TextBlob Sentiment
        tb = TextBlob(text).sentiment
        sentiments.append({
            'vader_neg': vs['neg'],
            'vader_neu': vs['neu'],
            'vader_pos': vs['pos'],
            'vader_compound': vs['compound'],
            'textblob_polarity': tb.polarity,
            'textblob_subjectivity': tb.subjectivity
        })
    sentiment_df = pd.DataFrame(sentiments)
    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

# Function to classify sentiment labels
def classify_sentiment(row):
    if row['vader_compound'] >= 0.05:
        return 'Positive'
    elif row['vader_compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to extract emojis
def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

# Function to perform topic modeling
def topic_modeling(texts, num_topics=5):
    st.info("Performing topic modeling...")
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        processed_texts.append(tokens)
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)
    return topics

# Modified emotion_analysis function using TreebankWordTokenizer
def emotion_analysis(texts):
    st.info("Analyzing emotions...")
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import TreebankWordTokenizer

    lemmatizer = WordNetLemmatizer()
    tokenizer = TreebankWordTokenizer()
    emotions_list = []

    # Load NRC Emotion Lexicon
    emotion_df = pd.read_csv(
        'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
        sep='\t',
        header=None,
        names=['word', 'emotion', 'association']
    )
    emotion_df = emotion_df[emotion_df['association'] == 1]
    emotion_dict = emotion_df.groupby('word')['emotion'].apply(list).to_dict()

    for text in tqdm(texts):
        tokens = tokenizer.tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
        emotion_counter = Counter()
        for token in tokens:
            if token in emotion_dict:
                emotion_counter.update(emotion_dict[token])
        emotions_list.append(emotion_counter)
    return emotions_list

# Visualization functions
def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment_label'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['green', 'blue', 'red'], ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Comments')
    st.pyplot(fig)

def plot_word_cloud(texts, title):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=1200, height=800, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

def plot_emotion_heatmap(emotion_data):
    emotion_df = pd.DataFrame(emotion_data).fillna(0)
    emotion_sums = emotion_df.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=emotion_sums.index, y=emotion_sums.values, ax=ax)
    ax.set_title('Overall Emotion Distribution')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def plot_sentiment_over_time(df):
    df['published_at'] = pd.to_datetime(df['published_at'])
    df = df.sort_values('published_at')
    df['vader_compound_smooth'] = df['vader_compound'].rolling(window=10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df['published_at'], df['vader_compound_smooth'], color='orange')
    ax.set_title('Sentiment Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Smoothed Compound Sentiment Score')
    st.pyplot(fig)

def plot_emoji_distribution(emojis):
    if not emojis:
        st.info("No emojis found.")
        return
    emoji_counts = Counter(emojis)
    most_common_emojis = emoji_counts.most_common(10)
    emojis_list, counts = zip(*most_common_emojis)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(emojis_list), y=list(counts), ax=ax)
    ax.set_title('Top 10 Emojis Used')
    ax.set_xlabel('Emoji')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Main function
def main():
    st.title("YouTube Comments Sentiment Analysis")
    st.write("Analyze sentiments, emotions, and topics in YouTube video comments.")

    # Add the attribution message at the top
    st.markdown("**Brought to you by Gaurav Kumar Jha**")

    # Input field for YouTube video URL
    video_url = st.text_input("Enter a YouTube video URL:")

    if video_url:
        try:
            video_id = extract_video_id(video_url)
            st.write(f"**Video ID:** {video_id}")
            comments_df = get_comments(video_id)

            if not comments_df.empty:
                # Display raw comments data
                if st.checkbox("Show raw comments data"):
                    st.subheader("Raw Comments Data")
                    st.dataframe(comments_df)

                # Sentiment Analysis
                comments_df = analyze_sentiment(comments_df)
                comments_df['sentiment_label'] = comments_df.apply(classify_sentiment, axis=1)

                # Emotion Analysis
                emotion_data = emotion_analysis(comments_df['text'])

                # Topic Modeling
                topics = topic_modeling(comments_df['text'])
                st.subheader("Identified Topics")
                for idx, topic in enumerate(topics):
                    st.write(f"**Topic {idx+1}:** {topic}")

                # Emoji Analysis
                comments_df['emojis'] = comments_df['text'].apply(extract_emojis)
                all_emojis = sum(comments_df['emojis'], [])

                # Word Frequency Analysis
                stop_words = set(stopwords.words('english'))
                all_words = []
                for text in comments_df['text']:
                    tokens = word_tokenize(text.lower())
                    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
                    all_words.extend(tokens)
                word_freq = Counter(all_words)
                most_common_words = word_freq.most_common(20)
                st.subheader("Most Common Words")
                common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
                st.table(common_words_df)

                # Display Results
                total_comments = len(comments_df)
                sentiment_counts = comments_df['sentiment_label'].value_counts()
                st.subheader("Sentiment Analysis Results")
                st.write(f"**Total comments analyzed:** {total_comments}")
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / total_comments) * 100
                    st.write(f"**{sentiment} comments:** {count} ({percentage:.2f}%)")
                st.write(f"**Total emojis found:** {len(all_emojis)}")

                # Visualizations
                st.subheader("Visualizations")
                plot_sentiment_distribution(comments_df)
                plot_word_cloud(comments_df['text'], 'Word Cloud of All Comments')
                plot_emotion_heatmap(emotion_data)
                plot_sentiment_over_time(comments_df)
                plot_emoji_distribution(all_emojis)

                # Display the closing attribution message
                st.markdown("---")  # Adds a horizontal line
                st.markdown("<p style='text-align: center;'>Made with ❤️ by Gaurav Kumar Jha</p>", unsafe_allow_html=True)


            else:
                st.warning("No comments found for the video.")
        except ValueError as e:
            st.error(e)

# Run the analysis
if __name__ == "__main__":
    main()
