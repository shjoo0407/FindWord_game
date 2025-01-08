import streamlit as st
import fasttext
import numpy as np
from transformers import BertTokenizer, BertModel

@st.cache_resource
def load_kobert():
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertModel.from_pretrained('monologg/kobert', trust_remote_code=True)
    return tokenizer, model


@st.cache_resource
def load_fasttext():
    model = fasttext.load_model('cc.ko.300.bin')  # FastText 모델 경로 설정
    return model


def get_kobert_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)

    # [CLS] 토큰과 마지막 레이어에서 나온 임베딩 벡터들 사용
    last_hidden_state = outputs.last_hidden_state

    # 해당 단어에 대응하는 임베딩 벡터만 추출
    word_embedding = last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    # 벡터 정규화 TODO:해야되는지는 모르겠음.
    norm_word_embedding = word_embedding / np.linalg.norm(word_embedding)

    return norm_word_embedding


def get_fasttext_embedding(word, fasttext_model):
    word_embedding = fasttext_model.get_word_vector(word)

    # 벡터 정규화
    norm_word_embedding = word_embedding / np.linalg.norm(word_embedding)

    return norm_word_embedding