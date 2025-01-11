import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

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

pg = st.navigation({
    "Main": [st.Page("fake_news_detection/homepage.py", title="🏠 Homepage")],
    "Information": [st.Page("fake_news_detection/information.py", title="📝 Tools Information")],
    "Tools": [st.Page("fake_news_detection/tools.py", title="🛠️ Fake News Detection")]
    })

pg.run()
