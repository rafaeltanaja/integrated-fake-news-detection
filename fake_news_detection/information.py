import streamlit as st 

st.title('Fake News Detection: FakeBERT Classification, Style Analysis, and Credibility Verification')
st.markdown("""
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    <div class="justified-text">
        Welcome to our Fake News Detection platform, 
        where cutting-edge technology meets critical analysis. Dive into our innovative 
        solutions like FakeBERT classification, style analysis, and credibility verification to combat misinformation and 
        promote trust in digital news content.
    </div>
    <br>
    """, unsafe_allow_html=True)

st.image("./assets/fake_news_detection_approach.png", caption="Fake News Detection Approach")

st.header('FakeBERT Classification')
st.markdown("""
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    <div class="justified-text">
        The FakeBERT Classification model utilizes a pre-trained BERT model combined with a fully connected layer, 
        trained on the ISOT fake news dataset, to enhance news text classification. The model achieves an accuracy of 88%.
    </div>
    <br>
    """, unsafe_allow_html=True)

st.header('Style Analysis')
st.markdown("""
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    <div class="justified-text">
        <p>Style Analysis involves calculating text content and representing it as numeric metrics. These metrics are categorized into three key dimensions:</p>
        <ul>
            <li><strong>Lexical Diversity Rate:</strong> The proportion of unique words in the text.</li>
            <li><strong>Spell Score Rate:</strong> The percentage of words with spelling errors.</li>
            <li><strong>Sentiment Score Rate:</strong> The overall polarity of the text, indicating positivity or negativity.</li>
        </ul>
    </div>
    <br>
    """, unsafe_allow_html=True)

st.header('Credibility Verification')
st.markdown("""
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    <div class="justified-text">
    Credibility Verification is an approach that leverages the title and date to identify related news from a trusted website, 
    <a href="https://today.yougov.com/politics/articles/49552-trust-in-media-2024-which-news-outlets-americans-trust" target="_blank">
        YouGov Trust in Media 2024
    </a>. 
    The cosine similarity between the three related news articles is then calculated, and a dynamic weighted mean is applied to determine the final results.
    </div>
    <br>
    """, 
    unsafe_allow_html=True)