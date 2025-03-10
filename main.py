import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
import os
import numpy as np
from transformers import pipeline
import librosa
import librosa.display
from nrclex import NRCLex
import re
import nltk 

for resource, path in [('punkt', 'tokenizers/punkt'), ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

st.title("Sentiment Analysis")

st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Select Input Type:", ("Text", "File Upload", "Audio Upload"))
analysis_mode = st.sidebar.radio("Select Analysis Mode:", ("Basic", "Advanced"))
language = "English"


def preprocess_text(text, lang):
    if lang != "English":
        try:
            text_translated = str(TextBlob(text).translate(to='en'))
            return text_translated
        except Exception as e:
            st.warning(f"Translation failed: {e}. Using original text.")
            return text
    else:
        return text

def analyze_sentiment_basic(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

@st.cache_resource(show_spinner=False)
def load_advanced_pipeline():
    return pipeline("sentiment-analysis")

def analyze_sentiment_advanced(text, advanced_analyzer):
    result = advanced_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

def analyze_emotion(text):
    emotion = NRCLex(text)
    return emotion.top_emotions  # Returns a list of (emotion, score)

def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

def plot_audio_features(audio_path):
    y, sr_rate = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # Plot waveform
    librosa.display.waveshow(y, sr=sr_rate, ax=ax[0])
    ax[0].set_title("Waveform")
    # Plot Mel Spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr_rate)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr_rate, x_axis='time', y_axis='mel', ax=ax[1])
    ax[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")
    st.pyplot(fig)

def classify_sentiment(polarity):
    """Classify sentiment into Positive, Negative, or Neutral based on polarity."""
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --- NEW: Enhanced Word Cloud Function ---
def generate_segregated_wordcloud(text):
    """
    Generate segregated word clouds for Positive, Negative, and Neutral words.
    The text is tokenized and each word's polarity is computed using TextBlob.
    """
    # Tokenize the text into words (ignoring punctuation)
    words = re.findall(r'\w+', text)
    
    # Build a frequency dictionary for all words (in lowercase)
    freq_dict = {}
    for word in words:
        w = word.lower()
        freq_dict[w] = freq_dict.get(w, 0) + 1
    
    # Create separate dictionaries for each sentiment category
    positive_dict = {}
    negative_dict = {}
    neutral_dict = {}
    
    for word, freq in freq_dict.items():
        polarity = TextBlob(word).sentiment.polarity
        if polarity > 0.1:
            positive_dict[word] = freq
        elif polarity < -0.1:
            negative_dict[word] = freq
        else:
            neutral_dict[word] = freq
            
    # Create a subplot with three columns for the three categories
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Positive words word cloud
    if positive_dict:
        wc_positive = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(positive_dict)
        axes[0].imshow(wc_positive, interpolation='bilinear')
        axes[0].axis("off")
        axes[0].set_title("Positive Words")
    else:
        axes[0].text(0.5, 0.5, "No Positive Words", horizontalalignment='center',
                     verticalalignment='center', fontsize=12)
        axes[0].axis("off")
        axes[0].set_title("Positive Words")
    
    # Negative words word cloud
    if negative_dict:
        wc_negative = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(negative_dict)
        axes[1].imshow(wc_negative, interpolation='bilinear')
        axes[1].axis("off")
        axes[1].set_title("Negative Words")
    else:
        axes[1].text(0.5, 0.5, "No Negative Words", horizontalalignment='center',
                     verticalalignment='center', fontsize=12)
        axes[1].axis("off")
        axes[1].set_title("Negative Words")
    
    # Neutral words word cloud
    if neutral_dict:
        wc_neutral = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(neutral_dict)
        axes[2].imshow(wc_neutral, interpolation='bilinear')
        axes[2].axis("off")
        axes[2].set_title("Neutral Words")
    else:
        axes[2].text(0.5, 0.5, "No Neutral Words", horizontalalignment='center',
                     verticalalignment='center', fontsize=12)
        axes[2].axis("off")
        axes[2].set_title("Neutral Words")
    
    plt.tight_layout()
    return fig

if input_type == "Text":
    user_text = st.text_area("Enter Text Below:", "")
    analyze_button = st.button("Analyze Sentiment")
elif input_type == "File Upload":
    uploaded_file = st.file_uploader("Upload a Text File (TXT or CSV):", type=["txt", "csv"])
    analyze_button = st.button("Analyze File")
elif input_type == "Audio Upload":
    uploaded_audio = st.file_uploader("Upload an Audio File (WAV):", type=["wav"])
    analyze_button = st.button("Analyze Audio")


if analyze_button:
    if input_type == "Text":
        if user_text.strip():
            processed_text = preprocess_text(user_text, language)
            st.write("### Processed Text for Analysis")
            st.write(processed_text)
            
            if analysis_mode == "Basic":
                polarity, subjectivity = analyze_sentiment_basic(processed_text)
                st.write("#### Sentiment Results (Basic Analysis)")
                st.write(f"**Polarity:** {polarity:.2f}")
                st.write(f"**Subjectivity:** {subjectivity:.2f}")
                sentiment_label = classify_sentiment(polarity)
                st.write(f"**Sentiment Category:** {sentiment_label}")
            else:
                advanced_analyzer = load_advanced_pipeline()
                label, score = analyze_sentiment_advanced(processed_text, advanced_analyzer)
                st.write("#### Sentiment Results (Advanced Analysis)")
                st.write(f"**Label:** {label}")
                st.write(f"**Confidence Score:** {score:.2f}")

            emotions = analyze_emotion(processed_text)
            st.write("#### Detected Emotions")
            for emo, emo_score in emotions:
                st.write(f"{emo.capitalize()}: {emo_score}")

            st.write("#### Enhanced Word Cloud (Segregated by Sentiment)")
            fig = generate_segregated_wordcloud(processed_text)
            st.pyplot(fig)
        else:
            st.error("Please enter some text to analyze.")

    elif input_type == "File Upload":
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.txt'):
                    text = uploaded_file.read().decode("utf-8")
                    df = pd.DataFrame({'Text': [text]})
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    raise ValueError("Unsupported file format.")
                
                if 'Text' not in df.columns:
                    st.error("CSV file must contain a 'Text' column.")
                else:
                    sentiment_results = []
                    sentiment_labels = []
                    advanced_analyzer = load_advanced_pipeline() if analysis_mode == "Advanced" else None

                    for txt in df['Text']:
                        processed_txt = preprocess_text(txt, language)
                        if analysis_mode == "Basic":
                            polarity, subjectivity = analyze_sentiment_basic(processed_txt)
                            sentiment_results.append((polarity, subjectivity))
                            sentiment_labels.append(classify_sentiment(polarity))
                        else:
                            label, score = analyze_sentiment_advanced(processed_txt, advanced_analyzer)
                            sentiment_results.append((label, score))
                            sentiment_labels.append(label)
                    
                    if analysis_mode == "Basic":
                        df[['Polarity', 'Subjectivity']] = pd.DataFrame(sentiment_results, index=df.index)
                    else:
                        df[['Label', 'Confidence']] = pd.DataFrame(sentiment_results, index=df.index)
                    
                    df['Sentiment'] = sentiment_labels
                    st.write("### File Sentiment Results")
                    st.write(df)

                    if analysis_mode == "Basic":
                        st.write("#### Sentiment Distribution")
                        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                        df['Polarity'].plot(kind='hist', bins=20, ax=ax[0], color='skyblue', title="Polarity Distribution")
                        ax[0].set_xlabel("Polarity")
                        df['Subjectivity'].plot(kind='hist', bins=20, ax=ax[1], color='salmon', title="Subjectivity Distribution")
                        ax[1].set_xlabel("Subjectivity")
                        sentiment_counts = df['Sentiment'].value_counts()
                        ax[2].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
                        ax[2].set_title("Sentiment Categories")
                        st.pyplot(fig)

                    st.write("#### Enhanced Word Cloud (Segregated by Sentiment)")
                    all_text = " ".join(df['Text'].tolist())
                    processed_all_text = preprocess_text(all_text, language)
                    fig = generate_segregated_wordcloud(processed_all_text)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.error("Please upload a file to analyze.")

    elif input_type == "Audio Upload":
        if uploaded_audio is not None:
            try:
                temp_audio_path = "temp_audio.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_audio.read())

                st.write("#### Audio Features")
                plot_audio_features(temp_audio_path)

                text_from_audio = audio_to_text(temp_audio_path)
                os.remove(temp_audio_path)
                
                processed_text = preprocess_text(text_from_audio, language)
                st.write("#### Extracted Text from Audio")
                st.write(processed_text)
                
                if analysis_mode == "Basic":
                    polarity, subjectivity = analyze_sentiment_basic(processed_text)
                    st.write("#### Sentiment Results from Audio (Basic Analysis)")
                    st.write(f"**Polarity:** {polarity:.2f}")
                    st.write(f"**Subjectivity:** {subjectivity:.2f}")
                    sentiment_label = classify_sentiment(polarity)
                    st.write(f"**Sentiment Category:** {sentiment_label}")
                else:
                    advanced_analyzer = load_advanced_pipeline()
                    label, score = analyze_sentiment_advanced(processed_text, advanced_analyzer)
                    st.write("#### Sentiment Results from Audio (Advanced Analysis)")
                    st.write(f"**Label:** {label}")
                    st.write(f"**Confidence Score:** {score:.2f}")

                emotions = analyze_emotion(processed_text)
                st.write("#### Detected Emotions")
                for emo, emo_score in emotions:
                    st.write(f"{emo.capitalize()}: {emo_score}")

                st.write("#### Enhanced Word Cloud (Segregated by Sentiment)")
                fig = generate_segregated_wordcloud(processed_text)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing audio: {e}")
        else:
            st.error("Please upload an audio file to analyze.")