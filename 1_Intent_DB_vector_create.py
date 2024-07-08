from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import openai, os
import json
import numpy as np
from scipy.spatial.distance import cosine

load_dotenv(verbose=False)

from openai import OpenAI
client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),  # this is also the default, it can be omitted
)

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    print(text)
    return response.data[0].embedding

def vectorize_gold_intent(gold_intent):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    vector = get_embedding(gold_intent)
    return vector if vector else []

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for i, entry in enumerate(data):
        gold_intent = entry.get("GOLD_INTENT", "")
        vectorized_intent = vectorize_gold_intent(gold_intent)
        entry["GOLD_INTENT_VECTOR"] = vectorized_intent
        data[i] = entry
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

#통합된 스토리1-intent-스토리2 json을 입력받음
input_json_path = '/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_text/merged_DB_text.json'
#intent를 벡터화해서 저장
output_json_path = '/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_vector/merged_DB_vector.json'
process_json(input_json_path, output_json_path)

#요거를 잘 반복해서 통합 벡터 db를 만들어야함.
