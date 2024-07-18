import os
import json
from generate import Generator

Intent_augment = Generator("chatopenai_4o", 
                            0.1, 
                            instruction_path='./instructions/augment_intent_from_base_story.txt')

def process_json_files(directory, output_path):
    count = 1
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            for i in range(4): 
                storyT0 = data.get(f"story{i}", "") 
                storyT1 = data.get(f"story{i+1}", "")
                intent = data.get(f"intent{i}", "") 
                data[f"intent{i}"] = Intent_augment.generate({'storyT' : storyT0, 'storyT+1' : storyT1, 'intent' : intent})
                print(f"Generated intent{i}: {data[f'intent{i}']}")  
                
            # Save the updated JSON file
            output_filepath = os.path.join(output_path, f'{count}.json')
            with open(output_filepath, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Saved file: {output_filepath}") 
            count += 1
            print(f"Processed file count: {count}") 

# Path to the directory containing the JSON files
directory_path = "./base_story_DB"
output_path = './base_story_intent_augmented_DB'
process_json_files(directory_path, output_path)
