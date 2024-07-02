import openai, json
import tiktoken

# GPT-4 모델을 위한 엔코더 불러오기
enc = tiktoken.encoding_for_model("gpt-4o")

with open('/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_vector/merged_DB_vector_filtered.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(f"원래 데이터 개수 : {len(data)}")
# instruction = 628 

intent_dict = {}
for entry in data:
    connected_story_T0 = entry.get("CHAPTER_T0", "")
    connected_story_T1 = entry.get("CHAPTER_T1", "")
    gold_intent = entry.get("GOLD_INTENT", "")
    
    if '.' in gold_intent:
        gold_intent=gold_intent[:-1]
    
    if gold_intent not in intent_dict:
        intent_dict[gold_intent] = 1
    else:
        intent_dict[gold_intent] += 1

all = []
for i in intent_dict:
    all.append((i, intent_dict[i]))

sorted_all = sorted(all, key=lambda x:x[1], reverse=True)



story_list = [
    "A story that justifies the current narrative through past recollections",
    "A story that explains a character's current actions through their past",
    "A story that provides a break to relieve tension",
    "A story that heightens anticipation for the next chapter",
    "A story that clarifies the conflict between opposing characters",
    "A story that guides the reader to solve the mystery alongside the characters",
    "A story that provides new information to the reader",
    "A story that elicits emotional immersion from the reader",
    "A story that prepares a twist contrary to the reader's expectations",
    "A story that hints at future events",
    "A story that subverts the reader's expectations through a twist",
    "A story that plants a trigger for a potential twist",
    "A story that reinforces the main plot through relatively less important events",
    "A story that delves deeper into and describes the background of the event",
    "A story that shows a character's evolution through the aftermath of an event",
    "A story that compares contrasting perspectives",
    "A story that deepens the theme through symbolic events",
    "A story that heightens tension by introducing new conflicts",
    "A story that introduces a new place or environment",
    "A story that complements the main plot through a subplot",
    "A story that expands the world-building to aid the reader's understanding",
    "A story that stimulates curiosity by introducing a mysterious element",
    "A story that emphasizes psychological conflict",
    "A story that creates a sense of urgency by setting up a crisis",
    "A story that changes the mood through humor",
    "A story that highlights the central conflict of the narrative",
    "A story that reveals the meaning of the foreshadowing mentioned in the previous chapter",
    "A story that resolves the mystery from the previous chapter",
    "A story that shows the consequences of actions from the previous chapter",
    "A story that progresses the plot through a turning point",
    "A story that shows a character's reaction to a major event",
    "A story that provides a new perspective by reexamining a major event",
    "A story that progresses the plot by showing the outcome of a major event",
    "A story that explores the aftermath of a major event",
    "A story that highlights elements that could lead to a major event",
    "A story that develops relationships between main characters",
    "A story that explores a main character's backstory",
    "A story that emphasizes the main theme",
    "A story that clarifies the protagonist's motivations",
    "A story that exposes the protagonist's weaknesses",
    "A story that uses metaphors related to the theme",
    "A story that maximizes conflict between characters",
    "A story that hints at conflicts between characters",
    "A story that becomes a catalyst to test a character's resolve",
    "A story that explores the inner changes of a character",
    "A story that shows a character's growth",
    "A story that emphasizes a character's weaknesses",
    "A story that provides important background information through a flashback",
    "A story that hints at the plot's direction",
    "A story that makes the plot unpredictable by changing its direction"
]



cnt = 0
for intent, count in sorted_all:
    if count >= 2:
        if intent not in story_list:
            cnt += 1
            print(cnt, intent, count)
