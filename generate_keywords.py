import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os


data_name = './dataset_longlamp/abstract_generation_user/test-00000-of-00002'
if os.path.exists(f'{data_name}-keywords.parquet'):
    df = pd.read_parquet(f'{data_name}-keywords.parquet')
else:
    df = pd.read_parquet(f'{data_name}.parquet')

api_key = 'api_key'
base_url = 'https://api.zhizengzeng.com'
model_name = 'deepseek-chat'
counter = 0
from openai import OpenAI
client = OpenAI(api_key=api_key, base_url=base_url)

def call_api(prompt):
    global counter
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ])
    counter += 1
    return completion.choices[0].message.content


keyword_extraction_prompt = "Mention 5 short keywords of the following abstract of the paper that shows their main findings and claims (direct output keywords without any other formats): '{}'"

for index, row in df.iterrows():
    num_profiles = len(row['profile'])
    if not row['profile'][0]['keywords'] or 'keywords' not in row['profile'][0]:
        for profile in tqdm(row['profile']):
            prompt = keyword_extraction_prompt.format(profile['abstract'])
            profile['keywords'] = call_api(prompt)

        df.to_parquet(f'{data_name}-keywords.parquet')
        print(f"Saved at {index}, Total {counter} calls made")
    else:
        print(f"Skipping {index}")
    



