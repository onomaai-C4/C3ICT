
from dotenv import load_dotenv
import os
load_dotenv(verbose=False)
import random
from datetime import datetime
from typing import Union, List, Dict
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain_anthropic import ChatAnthropic
import transformers
import datasets
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import pipeline
import pandas as pd
import numpy as np
import json
import shutil
import os, time
import datetime
from transformers.generation.utils import GenerationConfig
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
tqdm.pandas()
import subprocess
from langchain_core.runnables import Runnable
from typing import Optional
from langchain_core.runnables import Runnable, RunnableConfig

GPT4_O_ARGS = {
    "model_name": "gpt-4o-2024-05-13",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "max_tokens": 2048,
}

class Generator:
    def __init__(self, 
                #  vectorstore_path:str = "./chroma_db",
                 model:str = "./model", 
                 #사용자 정의 모델인 경우 상대경로 입력
                 temperature:float = 0.1,
                 instruction_path:str = '',
                 device = 'cuda:0',
                 ):
        
        self.model = model

        if self.model == "chatopenai_4o":
            self.llm = ChatOpenAI(**GPT4_O_ARGS, temperature=temperature)
    
        print(f"Model Loaded : {self.llm}")
        
        with open(instruction_path, "r") as f:
            TEMPLATE = f.read()
    
        self.template = PromptTemplate.from_template(TEMPLATE)

        self.chain = self.template | self.llm

    def generate(self, input_data: Dict[str, Union[str, int, float]]) -> Union[str, List[Union[str, Dict]]]:
        with get_openai_callback() as cb:
            final_input = self.template.format(**input_data)
            
            
            if self.model in ["chatopenai", "chatanthropic", "chatanthropic_o", "chatanthropic_h", "chatopenai_4o", "chatanthropic_s"]:
                result = self.chain.invoke(input_data).content.replace('\"', ' ').replace('\'', ' ')
            else:
                result = self.chain.invoke(input_data).replace('\"', ' ').replace('\'', ' ')
                
            print(f'vvvvvvvv Below is the text that went into the LLM input vvvvvvvvv\n{final_input}\n')
            print(f'^^^^^^^^ Above is the text that went into the LLM input ^^^^^^^^\n')
            print(f'vvvvvvvv Below is the LLM output vvvvvvvvv\n{result}\n')
            print('^^^^^^^^ Above is the LLM output ^^^^^^^^\n')
            print(f'vvvvvvvv Cost vvvvvvvv\n{cb}\n^^^^^^^^^^^^^^^^^^')
            
            return result

if __name__ == "__main__":
    model_name = "chatopenai_4o"
    generator = Generator(model_name, 0.1, './instructions/get_intent_between_2story.txt')
    response = generator.generate({'chapter_0': '안녕', 'chapter_1': '으아'})
 
  
