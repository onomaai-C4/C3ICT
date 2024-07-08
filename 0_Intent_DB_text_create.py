from generate import Generator 
import requests, re, json, os

def get_bookcontent_from_url(url):
        now_book = requests.get(url).text
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        start_matches = list(re.finditer(start_pattern, now_book))
        end_matches = list(re.finditer(end_pattern, now_book))
        for start, end in zip(start_matches, end_matches):
            nowbook_content = now_book[start.end():end.start()].strip()
        return nowbook_content
    
#######################################33
import requests, re
from itertools import groupby

def reduce_spaces(data):
    return [key for key, group in groupby(data, key=lambda x: x == ' ') for _ in ([key] if not key else [' '])]

def get_bookcontent_from_url(url):
        now_book = requests.get(url).text
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        start_matches = list(re.finditer(start_pattern, now_book))
        end_matches = list(re.finditer(end_pattern, now_book))
        for start, end in zip(start_matches, end_matches):
            nowbook_content = now_book[start.end():end.start()].strip()
        return nowbook_content

def chunk_text(text):
    chapter_spliter_count = text.count("CHAPTER")
    chunks = []
    chapters = text.split("CHAPTER")
    for i in range(len(chapters)):
        if len(chapters[i]) > 300:
            chunks.append("CHAPTER"+chapters[i])

    if len(chunks) == chapter_spliter_count:
        #컨텐츠 테이블이 없는 책
        return chunks
    else :
        return chunks[1:]
#######################################

if __name__ == "__main__":
    def get_url_from_num(num):
        return f'https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt'
    
    book_url_num_list = [ #상위 71
    145, 67979, 4085, 11, 2554, 174, 98, 76, 27827, 1260, 45, 74, 768, 244, 67098, 514, 158, 161, 829, 105, 521, 10676,
    164, 308, 41445, 863, 766, 203, 1837, 103, 121, 8117, 67138, 910, 500, 141, 2610, 599, 73771, 1023, 805, 73798, 113, 
    5230, 21816, 86, 18857, 73750, 42324, 61221, 4276, 1093, 140, 2559, 696, 6737, 62, 73917, 73824, 73780, 73727, 73787,
    1695, 26654, 73858, 963, 1056, 73734, 580, 1081, 974]

    book_url_list = [get_url_from_num(num) for num in book_url_num_list]#여따가 구텐베르그 책들 경로를 다 집어넣으면, 걔네들의 자연어 db(intent triplet)이 만들어짐 json파일로

    get_intent_module = Generator("chatopenai_4o", 0.1, instruction_path='/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/instructions/get_intent_between_2story.txt')
    
    
    for nowbook_url in book_url_list:
        nowbook_num = nowbook_url.split('/pg')[0].split('/')[-1]
        nowbook_data = []
        nowbook_TRIPLET_path = f'/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_text/{nowbook_num}_storyT0_INTENT_storyT1_TRIPLET.json'
        
        if not os.path.exists(nowbook_TRIPLET_path):
            with open(nowbook_TRIPLET_path, 'w') as file:
                json.dump([], file)  
                
        nowbook_content = get_bookcontent_from_url(nowbook_url)
        chunks = chunk_text(nowbook_content)
        
        for i in range(len(chunks)-1):
            chapter_T0 = chunks[i] 
            chapter_T1 = chunks[i+1] 
            intent = get_intent_module.generate({'chapter_0': chapter_T0, 'chapter_1': chapter_T1})
            
            nowbook_data.append({'book_url' : nowbook_url,'CHAPTER_T0': chapter_T0,'GOLD_INTENT': intent,'CHAPTER_T1': chapter_T1})
            
            with open(nowbook_TRIPLET_path, 'w') as file:
                json.dump(nowbook_data, file, indent=4)
            
        # Specify the directory containing the JSON files
    directory = '/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_text'  # Replace with the path to your directory

    # Initialize an empty list to hold all data
    all_data = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Open and read the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Add the data to the list
                all_data.extend(data)

    # Save the merged data to a new JSON file
    with open('/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/intent_DB_text/merged_DB_text.json', 'w', encoding='utf-8') as file_merged:
        json.dump(all_data, file_merged, ensure_ascii=False, indent=4)

    print("Merged JSON has been saved to 'merged_DB_text.json'")

        