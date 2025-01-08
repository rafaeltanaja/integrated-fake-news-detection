import pandas as pd
import re
import streamlit as st
import numpy as np
import streamlit as st
import pickle
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from assets import processing

class BERT_Arch(torch.nn.Module):
    def __init__(self, bert_model_name):
        super(BERT_Arch, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, **inputs):
        # Access input IDs and attention mask from the inputs dictionary
        sent_id = inputs.get('input_ids')
        mask = inputs.get('attention_mask')

        # Pass input IDs and attention mask to the BERT model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)
        return x

with open('model/random_forest_model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

st.markdown(
    """
    <style>
    body {
        background-color: #FAF9F6; /* Light gray background */
    }
    div.stButton > button:first-child {
        background-color: green;
        color: white;
    }
    .true-class {
        color: green;
    }
    .fake-class {
        color: red;
    }
    </style>
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fake News Detection")
st.write("Our fake news detection system leverages the power of three integrated approaches to provide accurate and reliable predictions for English-language content. By combining advanced machine learning algorithms, linguistic analysis, and contextual verification, we ensure a robust solution to identify and combat misinformation effectively.")
user_input_title = st.text_area("Input news title:", height=100)
user_input_text = st.text_area("Input news text:", height=200)

# Create the DataFrame
df = pd.DataFrame({'title': [user_input_title], 'text': [user_input_text], 'date': [" "] })

if st.button("Continue"):
    with st.spinner("Processing..."):
        # https://today.yougov.com/politics/articles/49552-trust-in-media-2024-which-news-outlets-americans-trust

        columns_to_select = [
            'dynamic_weighted_mean_similarity',
            'spell_score',
            'lexical_diversity_rate',
            'sentiment_score',
            'fake_bert_prediction'
        ]

        final_enriched_data = processing.process(df)

        predictions = loaded_data.predict(final_enriched_data[columns_to_select])

        if predictions[0] == 'True':
            string = 'True'
        elif predictions[0] == 'Fake':
            string = 'Fake'

        st.header(f"Approach Results:")
        col1, col2, col3 = st.columns(3)
        
        col4, col5 = st.columns(2)
        
        col1.metric("FakeBERT Predictions", final_enriched_data['fake_bert_prediction'], border=True)
        col2.metric("Lexical Diversity Rate",  f"{final_enriched_data['dynamic_weighted_mean_similarity'].iloc[0]:.2f}", border=True)
        col3.metric("Spell Score Rate", f"{final_enriched_data['spell_score'].iloc[0]:.2f}", border=True)
        col4.metric("Sentiment Score Rate", f"{final_enriched_data['sentiment_score'].iloc[0]:.2f}", border=True)
        col5.metric("Weighted Mean Cosine Similarity", f"{final_enriched_data['dynamic_weighted_mean_similarity'].iloc[0]:.2f}", border=True)
        
        st.header(f"Final Results:")
        st.metric("Predictions", f"{string}", border=True)

        url_list = final_enriched_data[['scraped_news_1_url', 'scraped_news_2_url', 'scraped_news_3_url']].values.flatten().tolist()

        st.header("Related News:")
        
        for i, url in enumerate(url_list, 1):
            if pd.notna(url):
                st.write(url)
                st.metric("Cosine Similarity Score", f"{final_enriched_data[f'similarity_score{i}'].iloc[0]:.2f}", border=True)