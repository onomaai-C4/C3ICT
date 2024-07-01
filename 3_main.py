from generate import Generator 
import requests, re, json, os, openai
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine

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

get_relation_graph_nowstory = Generator("chatopenai_4o", 0.1, instruction_path='/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/instructions/get_relation_graph_nowstory.txt')
get_next_story_from_all_source = Generator("chatopenai_4o", 0.1, instruction_path='/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/instructions/get_next_story_from_all_source.txt')

accumulated_graph = []

def get_next_story_from_all_source_func(now_Story : str = '입력 문단',
                                   now_Intent : str = '이렇게이렇게해서다음문단쓰세요'):
    global accumulated_graph
    now_Story_relation_graph = get_relation_graph_nowstory.generate({'now_story' : now_Story})
    accumulated_graph.append(now_Story_relation_graph)
    
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
    
    with open('/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_vector/merged_DB_vector.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    top_match_Intent = None
    top_similarity = float('inf')  # Since cosine distance, lower is better
    for entry in data:
        gold_intent_vector = entry.get("GOLD_INTENT_VECTOR", [])
        if gold_intent_vector:
            similarity = cosine(user_intent_vector, gold_intent_vector)
            if similarity < top_similarity:
                top_similarity = similarity
                
                top_match_Intent = entry.get("GOLD_INTENT", "")
                connected_story_T0 = entry.get("CHAPTER_T0", "")
                connected_story_T1 = entry.get("CHAPTER_T1", "")
                
    I = connected_story_T0 + ('\n'+'-'*30+'\n') + top_match_Intent + ('\n'+'-'*30+'\n') + connected_story_T1
    
    next_story = get_next_story_from_all_source.generate({'now_story' : now_Story, 
                                                          'relation_accumulative' : R,
                                                          'user_intent' : now_Intent,
                                                          'storyT0_intent_storyT1_TRIPLET' : I})
    print("-"*50)
    print(f"유저입력 intent : {now_Intent}")
    print(f"유사도1등 gold intent : {top_match_Intent}")
    print(f"유사도 점수 : {top_similarity}")
    print("-"*50)
    return next_story


generated_data = []

user_initial_story = input("최초 스토리 문단 입력 : ")
print('-'*50)
now_story = user_initial_story
generated_data.append(now_story)
while True:
    now_intent = input("다음 스토리를 어떤 식으로 쓸깝숑? : ")
    next_story = get_next_story_from_all_source_func(now_Story=now_story, now_Intent=now_intent)
    print(next_story)
    generated_data.append(now_intent)
    generated_data.append(next_story)
    now_story = next_story
    # 의도를 통해서 다음 스토리를 만든 경우에만 정지명령 가능
    now_stop_question = input("이쯤에서 그만하고 저장할거면 y누르고, 계속해서 만들거면 다른키 누르면된단다. (y/n): ")
    if now_stop_question == 'y':
        break
    else:
        pass
    
# JSON에 저장할 딕셔너리 생성
output_dict = {}

# 'story'와 'intent'의 카운터 초기화
story_count = 0
intent_count = 0

# 리스트를 순서에 따라 'story'와 'intent'로 분류하여 딕셔너리에 저장
for index, item in enumerate(generated_data):
    if index % 2 == 0:  # 짝수 인덱스는 'story'
        key = f"story{story_count}"
        story_count += 1
    else:  # 홀수 인덱스는 'intent'
        key = f"intent{intent_count}"
        intent_count += 1
    output_dict[key] = item

from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("현재 시각:", formatted_time)

# JSON 파일로 저장
with open(f'/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/results/created_story_{formatted_time}.json', 'w') as json_file:
    json.dump(output_dict, json_file, indent=4)

print("JSON 파일이 저장되었습니다.")