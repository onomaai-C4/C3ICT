from generate import Generator
import requests, re, json, os, openai
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
import random
load_dotenv(verbose=False)

# 최초입력 INPUT : 첫 문단 + 그 다음 서술의도 
# R = 첫 문단의 관계 그래프 추출(누적)
# I = 서술의도와 유사한 storyT_intent_storuT+1 triplet을 검색
# INPUT + R + I -> 두번째 문단을 작성

# 둘째입력 INPUT : 두번째 문단 + 그 다음 서술의도 
# R = 첫 문단 두번째 문단의 관계 그래프 추출(누적)
# I = 서술의도와 유사한 storyT_intent_storuT+1 triplet을 검색
# INPUT + R + I -> 세번째 문단을 작성

# ... 마지막 문단까지 작성. 
# 유저 입력은 첫 문단 + 그 다음의 모든 서술의도들
# 출력은 완결된 하나의 스토리 

get_relation_graph_nowstory = Generator("chatopenai_4o", 0.1, instruction_path='./instructions/get_relation_graph_nowstory.txt')
get_next_story_from_all_source = Generator("chatopenai_4o", 0.1, instruction_path='./instructions/get_next_story_from_all_source.txt')

accumulated_graph = []

def get_next_story_from_all_source_func(now_Story : str = '입력 문단',
                                   now_Intent : str = '이렇게이렇게해서다음문단쓰세요'):
    global accumulated_graph
    now_Story_relation_graph = get_relation_graph_nowstory.generate({'now_story' : now_Story})
    
    accumulated_graph.append(now_Story_relation_graph)
    accumulated_graph.append('\n'+'-'*10+'\n')
    
    R = "".join(accumulated_graph)

    # Intent DB에서 gold_intent들 벡터화해서 저장해놓은 뒤, now_Intent랑 유사도 1등인 gold_intent의 triplet을 가져옴
    from openai import OpenAI
    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # this is also the default, it can be omitted
    )
    def get_embedding(text):
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    user_intent_vector = get_embedding(now_Intent)
    
    with open('./intent_DB_vector/merged_DB_vector.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    top_match_Intent = None
    top_similarity = float('inf')  # Since cosine distance, lower is better
    for entry in data:
        gold_intent_vector = entry.get("GOLD_INTENT_VECTOR", [])
        if gold_intent_vector: 
            similarity = cosine(user_intent_vector, gold_intent_vector)
            if similarity < top_similarity: #랜덤하게 가져오기 추가해야함
                top_similarity = similarity
                top_match_Intent = entry.get("GOLD_INTENT", "") # top 1 gold intent를 찾았음, 이제 이 intent로 연결된 페어중 하나를 랜덤하게 가져와야 함
              
    if '.' in top_match_Intent: #gpt4o가 뽑은 intent들 중 텍스트는 완전히 동일한데 .으로 끝나는 경우와 아닌 경우를 같은 것으로 취급
        top_match_Intent = top_match_Intent[:-1]
    else:
        top_match_Intent = top_match_Intent[:]

    story_triplet_candidate_list = []
    for entry in data:
        now_gold_intent_in_db = entry.get("GOLD_INTENT", "")
        if top_match_Intent in now_gold_intent_in_db : # 검색된 intent에서 구두점을 제거한 문장이, db의 원소들의 gold intent 문자열의 부분집합인 경우
            connected_story_T0 = entry.get("CHAPTER_T0", "")
            connected_story_T1 = entry.get("CHAPTER_T1", "")
            story_triplet_candidate_list.append((connected_story_T0,connected_story_T1)) 
            # 검색된 intent와 구두점 제외하고 동일한 gold intent를 가지는 스토리를 리스트에 저장, 이 리스트에서 랜덤한 하나를 뽑음
            
    
    print(f'Number of db datapoints corresponding to the selected gold intent: {len(story_triplet_candidate_list)}')
    (choiced_story_T0, choiced_story_T1) = random.choice(story_triplet_candidate_list) # 랜덤한 하나를 뽑음
                
    I = choiced_story_T0 + ('\n'+'-'*30+'\n') + top_match_Intent + ('\n'+'-'*30+'\n') + choiced_story_T1
    
    next_story = get_next_story_from_all_source.generate({'now_story' : now_Story, 
                                                          'relation_accumulative' : R,
                                                          'user_intent' : now_Intent,
                                                          'storyT0_intent_storyT1_TRIPLET' : I})
    print("-" * 50)
    print(f"User input intent: {now_Intent}")
    print(f"Top matching gold intent: {top_match_Intent}")
    print(f"Similarity score (cosine distance, lower is more similar): {top_similarity}")
    print("-" * 50)
    return next_story, top_match_Intent, R


generated_data = []

user_initial_story = input("Feel free to write the beginning of the story here! : ")
print('-'*50)
now_story = user_initial_story
generated_data.append(now_story)
while True:
    now_intent = input("INTENT - feel free to request any elements you want when writing the next story! : ")
    next_story, next_story_referenced_this_intent, next_story_referenced_this_relation = get_next_story_from_all_source_func(now_Story=now_story, now_Intent=now_intent)
    print(next_story)
    print('-'*100)
    generated_data.append(now_intent)
    generated_data.append(next_story_referenced_this_intent)
    generated_data.append(next_story_referenced_this_relation)
    generated_data.append(next_story)
    
    now_story = next_story
    # 의도를 통해서 다음 스토리를 만든 경우에만 정지명령 가능
    now_stop_question = input("If you want to stop here and save, press S. If you want to keep going, press any other key. (S)Save and Stop / continue : ")
    if now_stop_question == 's' or now_stop_question == 'S':
        break
    else:
        pass
    
# JSON에 저장할 딕셔너리 생성
output_dict = {}

# 'story'와 'intent'의 카운터 초기화
story_count = 0
intent_count = 0
refered_intent_count = 0
refered_relation_count = 0

for index, item in enumerate(generated_data):
    if index % 4 == 0:  # 4의 배수 인덱스는 'story'
        key = f"story{story_count}"
        story_count += 1
    elif index % 4 == 1:  # 4의 배수 + 1 인덱스는 'intent'
        key = f"intent{intent_count}"
        intent_count += 1
    elif index % 4 == 2:  # 4의 배수 + 2 인덱스는 'refered_intent'
        key = f"refered_intent{refered_intent_count}"
        refered_intent_count += 1
    elif index % 4 == 3:  # 4의 배수 + 3 인덱스는 'referenced_this_relation'
        key = f"refered_relation{refered_relation_count}"
        refered_relation_count += 1
    output_dict[key] = item

from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H_%M_%S")
print("현재 시각:", formatted_time)

# JSON 파일로 저장
with open(f'./results/created_story_{formatted_time}.json', 'w') as json_file:
    json.dump(output_dict, json_file, indent=4)

print("JSON 파일이 저장되었습니다.")
