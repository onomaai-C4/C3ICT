import os
import json

def replace_newlines_in_json(directory):
    # 지정된 디렉토리의 모든 파일을 순회
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            # JSON 파일을 염
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # 모든 값에서 '\n'을 ''로 대체하는 재귀 함수
            def replace_newlines(obj):
                if isinstance(obj, dict):
                    return {k: replace_newlines(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_newlines(elem) for elem in obj]
                elif isinstance(obj, str):
                    return obj.replace('\n', '')
                else:
                    return obj
            
            # 데이터에서 '\n'을 ''로 대체
            new_data = replace_newlines(data)
            
            # 변경된 내용을 동일한 파일에 저장
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(new_data, file, ensure_ascii=False, indent=4)

# 특정 폴더 경로를 지정
directory_path = "/data1/fabulator/GRAPH_STUDY/Relation_Intent_Story_Generation/AMT_data"
replace_newlines_in_json(directory_path)
