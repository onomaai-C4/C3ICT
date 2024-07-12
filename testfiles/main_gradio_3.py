import gradio as gr
from generate import Generator
import requests, re, json, os, openai
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
import random

# Assuming the Generator class and OpenAIEmbeddings are already defined/imported

load_dotenv(verbose=False)

# Initialize generators with appropriate paths
get_relation_graph_nowstory = Generator("chatopenai_4o", 0.1, instruction_path='/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/instructions/get_relation_graph_nowstory.txt')
get_next_story_from_all_source = Generator("chatopenai_4o", 0.1, instruction_path='/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/instructions/get_next_story_from_all_source.txt')

accumulated_graph = []
generated_data = []
user_initial_story = ""
ui_display_data = []  # Store data for UI display

def get_next_story_from_all_source_func(now_Story, now_Intent):
    global accumulated_graph
    now_Story_relation_graph = get_relation_graph_nowstory.generate({'now_story': now_Story})
    
    accumulated_graph.append(now_Story_relation_graph)
    accumulated_graph.append('\n' + '-'*10 + '\n')
    
    R = "".join(accumulated_graph)

    # Intent DB에서 gold_intent들 벡터화해서 저장해놓은 뒤, now_Intent랑 유사도 1등인 gold_intent의 triplet을 가져옴
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
            if similarity < top_similarity: #랜덤하게 가져오기 추가해야함
                top_similarity = similarity
                top_match_Intent = entry.get("GOLD_INTENT", "")
              
    if '.' in top_match_Intent:
        top_match_Intent = top_match_Intent[:-1]
    else:
        top_match_Intent = top_match_Intent[:]

    story_triplet_candidate_list = []
    for entry in data:
        now_gold_intent_in_db = entry.get("GOLD_INTENT", "")
        if top_match_Intent in now_gold_intent_in_db:
            connected_story_T0 = entry.get("CHAPTER_T0", "")
            connected_story_T1 = entry.get("CHAPTER_T1", "")
            story_triplet_candidate_list.append((connected_story_T0, connected_story_T1))
    
    (choiced_story_T0, choiced_story_T1) = random.choice(story_triplet_candidate_list)
                
    I = choiced_story_T0 + ('\n' + '-'*30 + '\n') + top_match_Intent + ('\n' + '-'*30 + '\n') + choiced_story_T1
    
    next_story = get_next_story_from_all_source.generate({
        'now_story': now_Story,
        'relation_accumulative': R,
        'user_intent': now_Intent,
        'storyT0_intent_storyT1_TRIPLET': I
    })

    return next_story, top_match_Intent, R

def add_initial_story(initial_story):
    global user_initial_story, ui_display_data
    user_initial_story = initial_story
    generated_data.append(user_initial_story)
    ui_display_data.append(user_initial_story)
    return "Initial story added. Now, provide the next intent.", "\n".join(ui_display_data), user_initial_story

def generate_next_story(intent):
    global user_initial_story, generated_data, ui_display_data
    if not user_initial_story:
        return "Please add the initial story first.", "", user_initial_story
    
    next_story, _, _ = get_next_story_from_all_source_func(now_Story=user_initial_story, now_Intent=intent)
    generated_data.append(intent)
    generated_data.append(next_story)
    
    ui_display_data.append(intent)
    ui_display_data.append(next_story)
    
    user_initial_story = next_story  # Update for next iteration
    
    return next_story, "\n".join(ui_display_data), user_initial_story

def save_story():
    output_dict = {}
    story_count = intent_count = 0

    for index, item in enumerate(generated_data):
        if index % 2 == 0:
            key = f"story{story_count}"
            story_count += 1
        elif index % 2 == 1:
            key = f"intent{intent_count}"
            intent_count += 1
        output_dict[key] = item

    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H_%M_%S")
    print("현재 시각:", formatted_time)

    file_path = f'/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/results/created_story_{formatted_time}.json'
    with open(file_path, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)

    return f"JSON file saved at {file_path}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Story Generator")
    initial_story_input = gr.Textbox(label="Initial Story")
    intent_input = gr.Textbox(label="Next Intent")
    next_story_output = gr.Textbox(label="Generated Story", interactive=False)
    accumulated_output = gr.Textbox(label="Accumulated Stories and Intents", interactive=False)
    add_initial_story_btn = gr.Button("Add Initial Story")
    generate_next_story_btn = gr.Button("Generate Next Story")
    save_story_btn = gr.Button("Save Story")

    add_initial_story_btn.click(add_initial_story, inputs=initial_story_input, outputs=[next_story_output, accumulated_output, initial_story_input])
    generate_next_story_btn.click(generate_next_story, inputs=intent_input, outputs=[next_story_output, accumulated_output, initial_story_input])
    save_story_btn.click(save_story, outputs=next_story_output)

demo.launch()
