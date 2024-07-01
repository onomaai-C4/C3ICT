import requests, re

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
    
    
if __name__ == "__main__":

    nowbook_content = get_bookcontent_from_url('https://www.gutenberg.org/cache/epub/60067/pg60067.txt')

    chunks = chunk_text(nowbook_content)
    for i, chunk in enumerate(chunks):
        print(chunk[:100])
        print('-'*50, len(chunk))
    print(len(chunks))
   
    