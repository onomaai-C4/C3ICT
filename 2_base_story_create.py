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



prefix_list = [
    "On a rainy evening, the old bookshop's door creaked open, revealing a hidden world.",
    "She found a mysterious key that promised to unlock forgotten secrets.",
    "The ancient tree in the forest whispered tales of long-lost magic.",
    "A letter arrived with no return address, just a single cryptic sentence inside.",
    "At the stroke of midnight, the town's clock tower began to chime an unfamiliar tune.",
    "The abandoned mansion on the hill was said to be haunted by more than just ghosts.",
    "He woke up with no memory of the previous day and a strange symbol on his hand.",
    "The new girl in town had an aura of mystery that no one could quite place.",
    "Deep in the attic, they discovered an old photograph that defied explanation.",
    "The carnival arrived overnight, bringing with it wonders and dangers alike.",
    "She stumbled upon a hidden door in her basement that wasn't there before.",
    "The stray cat seemed to understand more than any ordinary feline.",
    "An unexpected snowfall in the middle of summer signaled the beginning of a strange adventure.",
    "The painting in the museum started to move when no one was watching.",
    "An eerie melody played from the abandoned piano at dusk.",
    "He received an invitation to a party that didn't seem to exist.",
    "A lost diary revealed the secrets of a forgotten love story.",
    "The mirror in the antique shop reflected a world that wasn't their own.",
    "A sudden power outage plunged the city into darkness, but one house remained lit.",
    "She could hear voices through the old radio, even when it was turned off.",
    "The lighthouse beacon guided them to an island not on any map.",
    "He found a journal written in a language no one had ever seen before.",
    "The small town had a legend about a hidden treasure waiting to be found.",
    "A strange eclipse brought the hidden creatures of the night into view.",
    "She inherited a family heirloom that held a powerful secret.",
]

beginning_of_the_story = [
    "Craft the opening of the narrative.",
    "Develop the initial part of the tale.",
    "Compose the introduction of the story.",
    "Write the story’s starting point.",
    "Formulate the prologue of the narrative."
]

development_of_the_story = [
    "Build the middle section of the story.",
    "Elaborate on the story's progression.",
    "Expand on the main events of the tale.",
    "Write the story's unfolding.",
    "Develop the body of the narrative."
]

climax_of_the_story = [
    "Create the peak moment of the tale.",
    "Write the story's turning point.",
    "Describe the story's most intense part.",
    "Craft the story’s highest point of tension.",
    "Formulate the narrative's critical moment."
]

conclusion_of_the_story = [
    "Write the ending of the story.",
    "Develop the story’s resolution.",
    "Compose the conclusion of the narrative.",
    "Craft the final part of the tale.",
    "Conclude the story's events."
]


for prefix in prefix_list:
    
    generated_data = []
    
    user_initial_story = prefix
    print('-'*50)
    now_story = user_initial_story
    generated_data.append(now_story)
    
    for i in range(4):
        if i == 0:
            now_intent = random.choice(beginning_of_the_story)
            next_story = get_next_story_from_Only_Intent_func(now_Story=now_story, now_Intent=now_intent)
            print(next_story)
            generated_data.append(now_intent)
            generated_data.append(next_story)
            now_story = next_story
        elif i == 1:
            now_intent = random.choice(development_of_the_story)
            next_story = get_next_story_from_Only_Intent_func(now_Story=now_story, now_Intent=now_intent)
            print(next_story)
            generated_data.append(now_intent)
            generated_data.append(next_story)
            now_story = next_story
        elif i == 2:
            now_intent = random.choice(climax_of_the_story)
            next_story = get_next_story_from_Only_Intent_func(now_Story=now_story, now_Intent=now_intent)
            print(next_story)
            generated_data.append(now_intent)
            generated_data.append(next_story)
            now_story = next_story
        elif i == 3:
            now_intent = random.choice(conclusion_of_the_story)
            next_story = get_next_story_from_Only_Intent_func(now_Story=now_story, now_Intent=now_intent)
            print(next_story)
            generated_data.append(now_intent)
            generated_data.append(next_story)
            now_story = next_story
        
    
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