import gradio as gr
from generate import Generator
import requests, re, json, os, openai
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
import random
import tempfile

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
    print("-"*50)
    print(f"유저입력 intent : {now_Intent}")
    print(f"유사도1등 gold intent : {top_match_Intent}")
    print(f"유사도 점수(코사인 거리, 낮을수록 유사함) : {top_similarity}")
    print("-"*50)
    return next_story, top_match_Intent, R

def generate_next_story(initial_story, intent):
    global user_initial_story, generated_data, ui_display_data
    if not user_initial_story:
        user_initial_story = initial_story
        generated_data.append(user_initial_story)
        ui_display_data.append(user_initial_story)
    
    next_story, _, _ = get_next_story_from_all_source_func(now_Story=user_initial_story, now_Intent=intent)
    generated_data.append(intent)
    generated_data.append(next_story)
    
    ui_display_data.append(intent)
    ui_display_data.append(next_story)
    
    user_initial_story = next_story  # Update for next iteration
    
    return "\n".join(ui_display_data), user_initial_story, ""

def save_story(accumulated_output):
    file_name = f"{accumulated_output[:20]}.txt"
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(accumulated_output)
    
    return file_path
# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Story Generator")
    initial_story_input = gr.Textbox(label="Now Story")
    intent_input = gr.Textbox(label="Intent")
    accumulated_output = gr.Textbox(label="Accumulated Stories and Intents", interactive=False, elem_id="accumulated_output")
    generate_next_story_btn = gr.Button("Generate Next Story", elem_id="generate_next_story_btn")
    save_story_btn = gr.Button("Save Story")

    generate_next_story_btn.click(generate_next_story, inputs=[initial_story_input, intent_input], outputs=[accumulated_output, initial_story_input, intent_input])
    save_story_btn.click(save_story, inputs=accumulated_output, outputs=[gr.File(label="Download Story")])

demo.launch()
