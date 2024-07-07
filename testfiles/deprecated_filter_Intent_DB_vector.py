import openai, json
import tiktoken

# GPT-4 모델을 위한 엔코더 불러오기
enc = tiktoken.encoding_for_model("gpt-4o")

with open('/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_vector/merged_DB_vector.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(f"원래 데이터 개수 : {len(data)}")
# instruction = 628 

filtered_data = []

for entry in data:
    connected_story_T0 = entry.get("CHAPTER_T0", "")
    connected_story_T1 = entry.get("CHAPTER_T1", "")
    gold_intent = entry.get("GOLD_INTENT", "")

    tokens = enc.encode(connected_story_T0 + gold_intent + connected_story_T1)
    num_tokens = len(tokens)
    all_token_input = num_tokens + 628
    
    if all_token_input <= 8192:
        filtered_data.append(entry)

print(f"필터링 후 데이터 개수 : {len(filtered_data)}")
output_file_path = "/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_vector/merged_DB_vector_filtered.json"

# 새로운 JSON으로 저장
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
print(f"Filtered data has been saved to {output_file_path}")