from generate import Generator
import requests, re, json, os, openai
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
import random
load_dotenv(verbose=False)

# 최초입력 INPUT : 첫 문단 + 그 다음 서술의도 
# INPUT -> 두번째 문단을 작성

# 둘째입력 INPUT : 두번째 문단 + 그 다음 서술의도 
# INPUT -> 세번째 문단을 작성

# ... 마지막 문단까지 작성. 
# 유저 입력은 첫 문단 + 그 다음의 모든 서술의도들
# 출력은 완결된 하나의 스토리 

get_next_story_from_Only_Intent = Generator("chatopenai_4o", 0.1, instruction_path='/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/instructions/get_next_story_from_only_Intent.txt')

def get_next_story_from_Only_Intent_func(now_Story : str = '입력 문단',
                                   now_Intent : str = '이렇게이렇게해서다음문단쓰세요'):
    
    next_story = get_next_story_from_Only_Intent.generate({'now_story' : now_Story, 
                                                          'user_intent' : now_Intent})
    
    return next_story


generated_data = []

user_initial_story = input("최초 스토리 문단 입력 : ")
print('-'*50)
now_story = user_initial_story
generated_data.append(now_story)
while True:
    now_intent = input("다음 스토리를 어떤 식으로 쓸깝숑? : ")
    next_story = get_next_story_from_Only_Intent_func(now_Story=now_story, now_Intent=now_intent)
    print(next_story)
    generated_data.append(now_intent)
    generated_data.append(next_story)
    
    now_story = next_story
    # 의도를 통해서 다음 스토리를 만든 경우에만 정지명령 가능
    now_stop_question = input("이쯤에서 그만하고 저장할거면 S 누르고, 계속해서 만들거면 다른키 누르면된단다. Save and Stop / continue : ")
    if now_stop_question == 's' or now_stop_question == 'S':
        break
    else:
        pass
    
# JSON에 저장할 딕셔너리 생성
output_dict = {}

# 'story'와 'intent'의 카운터 초기화
story_count = 0
intent_count = 0

for index, item in enumerate(generated_data):
    if index % 2 == 0:  # 2의 배수 인덱스는 'story'
        key = f"story{story_count}"
        story_count += 1
    elif index % 2 == 1:  # 2의 배수 + 1 인덱스는 'intent'
        key = f"intent{intent_count}"
        intent_count += 1
    output_dict[key] = item

from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H_%M_%S")
print("현재 시각:", formatted_time)

# JSON 파일로 저장
with open(f'/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/base_story_DB/created_base_story_{formatted_time}.json', 'w') as json_file:
    json.dump(output_dict, json_file, indent=4)

print("JSON 파일이 저장되었습니다.")