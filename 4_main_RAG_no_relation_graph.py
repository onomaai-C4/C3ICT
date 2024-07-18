from generate import Generator
import requests, re, json, os, openai
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
import random
load_dotenv(verbose=False)

# 최초입력 INPUT : 첫 문단 + 그 다음 서술의도 
# I = 서술의도와 유사한 storyT_intent_storuT+1 triplet을 검색
# INPUT + I -> 두번째 문단을 작성

# 둘째입력 INPUT : 두번째 문단 + 그 다음 서술의도 
# I = 서술의도와 유사한 storyT_intent_storuT+1 triplet을 검색
# INPUT + I -> 세번째 문단을 작성

# ... 마지막 문단까지 작성. 
# 유저 입력은 첫 문단 + 그 다음의 모든 서술의도들
# 출력은 완결된 하나의 스토리 

get_next_story_from_all_source = Generator("chatopenai_4o", 0.1, instruction_path='./instructions/get_next_story_from_referenced_story.txt')


def get_next_story_from_all_source_func(now_Story : str = '입력 문단',
                                   now_Intent : str = '이렇게이렇게해서다음문단쓰세요'):

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
            
    
    print(f'뽑힌 gold intent에 해당하는 db데이터포인트 수 : {len(story_triplet_candidate_list)}')
    (choiced_story_T0, choiced_story_T1) = random.choice(story_triplet_candidate_list) # 랜덤한 하나를 뽑음
                
    I = choiced_story_T0 + ('\n'+'-'*30+'\n') + top_match_Intent + ('\n'+'-'*30+'\n') + choiced_story_T1
    
    next_story = get_next_story_from_all_source.generate({'now_story' : now_Story, 
                                                          'user_intent' : now_Intent,
                                                          'storyT0_intent_storyT1_TRIPLET' : I})
    print("-"*50)
    print(f"유저입력 intent : {now_Intent}")
    print(f"유사도1등 gold intent : {top_match_Intent}")
    print(f"유사도 점수(코사인 거리, 낮을수록 유사함) : {top_similarity}")
    print("-"*50)
    return next_story, top_match_Intent, I

base_story_intent_augmented_DB_path = './base_story_intent_augmented_DB'
count = 1 
for filename in os.listdir(base_story_intent_augmented_DB_path):

    if filename.endswith(".json"):
        filepath = os.path.join(base_story_intent_augmented_DB_path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        generated_data = []

        story0 = data.get(f"story0", "")  
        intent0 = data.get(f"intent0", "")  
        story1 = data.get(f"story1", "")  
        intent1 = data.get(f"intent1", "")  
        story2 = data.get(f"story2", "")  
        intent2 = data.get(f"intent2", "")  
        story3 = data.get(f"story3", "")  
        intent3 = data.get(f"intent3", "")  
        story4 = data.get(f"story4", "")  
 
        generated_data.append(story0) #원본 스토리0
        
        generated_data.append(intent0) #원본 intent0
        newstory1, retrieved_gold_intent0, retrieved_gold_triplet0 = get_next_story_from_all_source_func(story0, intent0)
        generated_data.append(retrieved_gold_intent0) #검색된 intent0
        generated_data.append(retrieved_gold_triplet0) #검색된 triplet0
        generated_data.append(newstory1) #만들어진 새 스토리1(원본 스토리1과 비교용, 다음 스텝에서 사용되지 않음)
        
        generated_data.append(intent1) #원본 intent1
        newstory2, retrieved_gold_intent1, retrieved_gold_triplet1= get_next_story_from_all_source_func(story1, intent1)
        generated_data.append(retrieved_gold_intent1) #검색된 intent1
        generated_data.append(retrieved_gold_triplet1) #검색된 triplet1
        generated_data.append(newstory2) #만들어진 새 스토리2(원본 스토리2와 비교용, 다음 스텝에서 사용되지 않음)
        
        generated_data.append(intent2) #원본 intent2
        newstory3, retrieved_gold_intent2, retrieved_gold_triplet2 = get_next_story_from_all_source_func(story2, intent2)
        generated_data.append(retrieved_gold_intent2) #검색된 intent2
        generated_data.append(retrieved_gold_triplet2) #검색된 triplet2
        generated_data.append(newstory3) #만들어진 새 스토리3(원본 스토리3과 비교용, 다음 스텝에서 사용되지 않음)
        
        generated_data.append(intent3) #원본 intent3
        newstory4, retrieved_gold_intent3, retrieved_gold_triplet3 = get_next_story_from_all_source_func(story3, intent3)
        generated_data.append(retrieved_gold_intent3) #검색된 intent3
        generated_data.append(retrieved_gold_triplet3) #검색된 triplet3
        generated_data.append(newstory4) #만들어진 새 스토리4(원본 스토리4와 비교용, 다음 스텝에서 사용되지 않음)
    
        # Note : 이 순서대로 generated_data에 삽입하는 이유는,
        # 생성된 단락이 intent를 지키면서 필력이 좋은지 평가하기 위한 원본과 비교하는 데이터셋을 구축하기 위함임. 
        # 생성된 스토리들끼리 이어지지 않는 것이 정상임에 주의.

        # JSON에 저장할 딕셔너리 생성
        output_dict = {}

        # 'story'와 'intent'의 카운터 초기화
        story_count = 0
        intent_count = 0
        retrieved_intent_count = 0
        retrieved_triplet_count = 0
    
        for index, item in enumerate(generated_data):
            if index % 4 == 0:  # 4의 배수 인덱스는 'story'
                key = f"story{story_count}"
                story_count += 1
            elif index % 4 == 1:  # 4의 배수 + 1 인덱스는 'intent'
                key = f"intent{intent_count}"
                intent_count += 1
            elif index % 4 == 2:  # 4의 배수 + 2 인덱스는 'retrieved_intent'
                key = f"retrieved_intent{retrieved_intent_count}"
                retrieved_intent_count += 1
            elif index % 4 == 3:  # 4의 배수 + 3 인덱스는 'retrieved_triplet'
                key = f"retrieved_triplet{retrieved_triplet_count}"
                retrieved_triplet_count += 1
                
            output_dict[key] = item

        # JSON 파일로 저장
        with open(f'./base_story_RAG_results/{count}_RAG_created_story.json', 'w') as json_file:
            json.dump(output_dict, json_file, indent=4)

        print("JSON 파일이 저장되었습니다.")
        count += 1
       