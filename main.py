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