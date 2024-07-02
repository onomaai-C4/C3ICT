import json
with open('/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_vector/merged_DB_vector.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

dict = {}
count = 0
for entry in data:
    gold_intent_vector = entry.get("GOLD_INTENT", [])
    if '.' in gold_intent_vector : gold_intent_vector[:-1]

    if gold_intent_vector == 'A story that explores a main character s backstory':
        count += 1
        if count == 1:
            print(entry.get("CHAPTER_T0", ""))
            print('-'*100)
            print(entry.get("CHAPTER_T1", ""))
            break
        