import requests, re, json, os, openai

base_story_path = '/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/base_story_intent_augmented_DB'
intent_triplet_RAG_results_path = '/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/base_story_RAG_results'
AMT_data_path = '/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/AMT_data'


files_base = [f for f in os.listdir(base_story_path) if f.endswith('.json')]
files_RAG = [f for f in os.listdir(intent_triplet_RAG_results_path) if f.endswith('.json')]

for file_base in files_base:
    base_filename = os.path.splitext(file_base)[0]
    base_number = base_filename.split('_')[0]
    file_RAG = f"{base_number}_RAG_created_story.json" #이거 바꾸기 
    
    with open(os.path.join(base_story_path, file_base), 'r', encoding='utf-8') as f:
        data_base = json.load(f)
    
    with open(os.path.join(intent_triplet_RAG_results_path, file_RAG), 'r', encoding='utf-8') as f:
        data_RAG = json.load(f)
    print(file_base, file_RAG)
    
    
    # 새로운 구조로 통합
    combined_data = []
    for i in range(4):
        combined_entry = {
            f'base_story{i}': data_base.get(f'story{i}'),
            f'intent{i}': data_base.get(f'intent{i}'),
            f'base_story{i+1}': data_base.get(f'story{i+1}'),
            f'RAG_story{i+1}': data_RAG.get(f'story{i+1}')
        }
        combined_data.append(combined_entry)
    
    # 폴더 C에 저장
    output_filename = os.path.join(AMT_data_path, f"{base_filename}_combined.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"{base_filename}_combined.json 저장됨")