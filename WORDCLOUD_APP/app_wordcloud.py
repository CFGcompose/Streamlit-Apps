import pandas as pd
import numpy as np
from collections import Counter

import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords


import matplotlib.pyplot as plt


import streamlit as st


##################################################################################

st.title('Wordcloud Maker')


placeholder = st.empty()

text = placeholder.text_input('text')
click_clear = st.sidebar.button('Clear and start again', key=1)
if click_clear:
    text = placeholder.text_input('text', value='', key=1)

if text:
    st.info("Your text is successfully uploaded. You can now create your wordcloud")



color = st.color_picker('Pick A background color', '#1A3643')

if st.button('Make Wordcloud'):
    #result = name.title()
    tokens = nltk.wordpunct_tokenize(text)
    words = [w.lower() for w in tokens if w.isalpha()]
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    
    word_freq = Counter(filtered_words)
    word_Shakes = word_freq.most_common()
    word_Shakes = word_Shakes[0:100]
    word_cloud = WordCloud(background_color = color, width = 1000, height = 1000).generate_from_frequencies(dict(word_Shakes))
    
    fig= plt.figure(figsize=(10,8))
    plt.imshow(word_cloud)#interpolation='bilinear'
    plt.axis('off')
    st.pyplot(fig)






