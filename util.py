import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from model import get_kobert_embedding, get_fasttext_embedding


def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0]


def calculate_rank(input_embedding, word_list, model_option, tokenizer=None, kobert_model=None, fasttext_model=None):
    """입력된 단어가 유사도 기준으로 몇 번째인지 계산"""
    similarities = []

    for word in word_list:
        if model_option == "KoBERT":
            word_embedding = get_kobert_embedding(word, tokenizer, kobert_model)
        else:
            word_embedding = get_fasttext_embedding(word, fasttext_model)

        similarity = calculate_similarity(input_embedding, word_embedding)
        similarities.append((word, similarity))

    # 유사도 내림차순 정렬
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # 입력 단어의 순위 찾기
    for rank, (word, similarity) in enumerate(similarities, start=1):
        if word == st.session_state["target_word"]:
            return rank, similarities
    return None, similarities