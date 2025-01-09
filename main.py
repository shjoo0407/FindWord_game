import random
import streamlit as st
from word_list import WORD_LIST
from model import load_kobert, load_fasttext, get_kobert_embedding, get_fasttext_embedding
from util import calculate_similarity, calculate_rank

def main():
    st.title('단어 맞추기')
    st.write('한국어 단어를 입력하고 유사도를 기반으로 정답을 맞춰보세요.')

    # 모델 선택
    model_option = st.selectbox("모델을 선택하세요", ["FastText","KoBERT[점검 필요]"])
    if model_option == "FastText":
        fasttext_model = load_fasttext()
    else:
        tokenizer, model = load_kobert()

    # 정답 단어 설정
    if "target_word" not in st.session_state:
        st.session_state["target_word"] = random.choice(WORD_LIST)

    # 정답 변경 버튼
    if st.button("새로운 정답 생성"):
        st.session_state["target_word"] = random.choice(WORD_LIST)
        st.session_state["answers"] = []  # 기존 답변 기록 초기화

    target_word = st.session_state["target_word"]

    # 답변 기록 초기화
    if "answers" not in st.session_state:
        st.session_state["answers"] = []

    # 사용자 입력
    input_word = st.text_input("답변을 입력 : ")

    # '정답 보기' 버튼
    if st.button("정답 보기"):
        st.write(f"정답 단어는 '{target_word}'입니다.")

    if input_word:
        try:
            # 모델에 따라 다른 임베딩 추출
            if model_option == "KoBERT":
                target_embedding = get_kobert_embedding(target_word, tokenizer, model)
                input_embedding = get_kobert_embedding(input_word, tokenizer, model)
            else:  # FastText
                target_embedding = get_fasttext_embedding(target_word, fasttext_model)
                input_embedding = get_fasttext_embedding(input_word, fasttext_model)

            similarity = calculate_similarity(target_embedding, input_embedding)

            # 순위 계산
            rank, similarities = calculate_rank(
                input_embedding,
                WORD_LIST,
                model_option,
                tokenizer=tokenizer if model_option == "KoBERT" else None,
                kobert_model=model if model_option == "KoBERT" else None,
                fasttext_model=fasttext_model if model_option == "FastText" else None,
            )

            # 유사도 계산 및 출력
            st.write(f"유사도 : {similarity:.2f}")
            st.write(f"입력한 단어의 유사도 순위: **{rank}/{len(WORD_LIST)}**")

            # 정답 여부 확인
            if input_word == target_word:
                st.success("정답입니다! 🎉")
                st.write(f"정답 단어: '{target_word}'")
            else:
                st.warning("틀렸습니다! 다시 시도해보세요.")

            # 답변과 유사도를 기록
            st.session_state["answers"].append({"답변": input_word, "유사도": round(similarity,3), "순위": rank})
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

    # 유사도 순으로 답변 기록 테이블 출력
    if st.session_state["answers"]:
        st.subheader("LOG")

        # 유사도에 따라 내림차순 정렬
        sorted_answers = sorted(st.session_state["answers"], key=lambda x: x["유사도"], reverse=True)

        st.table(sorted_answers)


if __name__ == '__main__':
    main()
