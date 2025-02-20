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

# App Title
st.title("Sentiment Analysis")

# Sidebar Options
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Select Input Type:", ("Text", "File Upload", "Audio Upload"))
analysis_mode = st.sidebar.radio("Select Analysis Mode:", ("Basic", "Advanced"))
language = st.sidebar.selectbox("Select Language:", ["English"])

# ----- Helper Functions -----

def preprocess_text(text, lang):
    """Translate text to English if needed."""
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
    """Analyze sentiment using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

@st.cache_resource(show_spinner=False)
def load_advanced_pipeline():
    """Load the Hugging Face sentiment-analysis pipeline."""
    return pipeline("sentiment-analysis")

def analyze_sentiment_advanced(text, advanced_analyzer):
    """Analyze sentiment using a transformer model."""
    result = advanced_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score