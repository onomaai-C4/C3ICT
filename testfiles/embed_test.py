import openai, os
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import os
load_dotenv(verbose=False)

# 환경 변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

intentions = [
    # Hansel and Gretel
    "Creating tension by showing the situation of getting lost.",
    "Emphasizing sibling love and cooperation to evoke readers' emotions.",
    "Raising readers' expectations by presenting a clue of hope.",
    "Introducing fairy tale elements to spark readers' interest.",
    "Maintaining tension through the element of temptation.",
    "Surprising readers through a plot twist.",
    "Encouraging readers' emotional engagement by highlighting Hansel's courage and sacrifice.",
    "Maintaining tension through Gretel's fear.",
    "Increasing anticipation for the story's progression by planning an escape.",
    "Creating a sense of crisis through escalating conflict.",
    "Sparking readers' hope by suggesting the possibility of escape.",
    "Highlighting the protagonist's wisdom through practical preparations.",
    "Maintaining readers' interest by heightening tension during the escape attempt.",
    "Sustaining the story's continuity by presenting new challenges.",
    "Emphasizing the protagonist's wisdom and preparedness.",
    
    # Little Red Riding Hood
    "Helping readers understand by clearly stating the story's purpose.",
    "Adding tension to the story by introducing elements of conflict.",
    "Stimulating readers' anxiety by hinting at potential threats.",
    "Creating a sense of unease about future developments by emphasizing the protagonist's innocence.",
    "Increasing the story's interest by escalating conflict.",
    "Sustaining readers' interest through twists and disguises.",
    "Showing the possibility of resolving conflicts through Little Red Riding Hood's courage.",
    "Heightening tension through the wolf's deception.",
    "Encouraging readers' emotional engagement by placing the protagonist in danger.",
    "Presenting hope through the arrival of a rescuer.",
    
    # Cinderella
    "Evoking readers' sympathy by highlighting the protagonist's hardships.",
    "Creating anticipation by presenting a turning point in the story.",
    "Increasing the story's interest by adding elements of conflict.",
    "Suggesting the possibility of hope through magical elements.",
    "Maximizing readers' expectations through miracles and transformations.",
    "Stimulating readers' emotions by adding romantic elements.",
    "Maintaining readers' anticipation by hinting at a happy ending.",
    "Evoking emotions by emphasizing the protagonist's courage and independence.",
    "Showing the protagonist's growth through changes in social status.",
    "Maximizing readers' satisfaction by ultimately realizing justice."
]
# intentions = [
#     "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", 
#     "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry", 
#     "strawberry", "tangerine", "ugli fruit", "watermelon",  # 과일 이름 20종류
#     "table", "chair", "computer", "book", "pencil", "notebook", "phone", "keyboard", 
#     "mouse", "monitor", "lamp", "bag", "bottle", "clock", "shoe", "shirt", 
#     "car", "bike", "house", "window"  # 과일이 아닌 단어 20종류
# ]

    

from openai import OpenAI

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),  # this is also the default, it can be omitted
)

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
        # model="text-embedding-ada-002"
    )
    return response.data[0].embedding


all = intentions 
# 입력 문장

# print(len(get_embedding('안녕하세요')))
# exit()

vectors = []
for i in range(len(all)):
    now_embedding = get_embedding(all[i])
    print(len(now_embedding))
    import time
    time.sleep(0.2)
    vectors.append((all[i], now_embedding))
    
##############

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_sort(input_word_vector, word_vector_list):
    # 입력 벡터와 리스트 내 모든 벡터 간의 코사인 유사도를 계산
    input_vector = np.array(input_word_vector).reshape(1, -1)
    vectors = np.array([vector for word, vector in word_vector_list])
    
    similarities = cosine_similarity(input_vector, vectors)[0]
    
    # 유사도에 따라 단어를 높은 순으로 정렬
    sorted_word_vector_pairs = sorted(zip(word_vector_list, similarities), key=lambda x: x[1], reverse=True)
    
    # 결과 출력
    sorted_words = [(word, similarity) for (word, vector), similarity in sorted_word_vector_pairs]
    return sorted_words

word_vector_list = vectors

input_text = 'Maintain the flow of the story.'

input_word_vector = get_embedding(input_text)

# 함수 실행
sorted_words = cosine_similarity_sort(input_word_vector, word_vector_list)

# 결과 출력
for word, similarity in sorted_words:
    print(f"Word: {word}, Cosine Similarity: {similarity:.4f}")


words = [word for word, similarity in sorted_words]
similarities = [similarity for word, similarity in sorted_words]

import matplotlib.pyplot as plt

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.barh(words, similarities, color='skyblue')
plt.xlabel(f'Cosine Similarity - {input_text}')
plt.ylabel('Words')
plt.title('Cosine Similarity of Words')
plt.gca().invert_yaxis()  # y축 뒤집기
plt.grid(axis='x')

# 이미지 파일로 저장
plt.savefig('cosine_similarity_chart.png')

# 그래프 보여주기
plt.show()
    
    