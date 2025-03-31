import pandas as pd
import os
import sys

CHUNK_SIZE = 100
TASK = sys.argv[1]
model_name = sys.argv[2]
chunk_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
begin_index = chunk_index * CHUNK_SIZE
end_index = (chunk_index + 1) * CHUNK_SIZE - 1


task_to_prompts = {
    "topic": "Generate the content for a Reddit post {}",
    "review": "Generate the review text written by a reviewer who has given an overall rating of {} for a product with description {}. The summary of the review text is {}.",
    "abs": 'Generate an abstract for the title "{}" using the following items: {}.'
}

llama_task_to_prompts = {
    "topic": "Generate the content for a reddit post {}",
    "review": "Generate the review text written by a reviewer who has given an overall rating of {} for a product with description {}. The summary of the review text is {}.",
}

task_to_names = {
    "topic": "topic_writing_user",
    "abs": "abstract_generation_user",
    "review": "product_review_user"
}
data_name = f'./dataset_longlamp/{task_to_names[TASK]}/test-00000-of-00001'
if os.path.exists(data_name + '-nonPers.parquet'):
    df = pd.read_parquet(data_name + '-nonPers.parquet')
else:
    df = pd.read_parquet(data_name + '.parquet')

api_key = 'api_key'
base_url = 'https://api.openai.com'

counter = 0
from openai import OpenAI
client = OpenAI(api_key=api_key, base_url=base_url)

from transformers import AutoModelForCausalLM, AutoTokenizer
def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    # device map auto
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    return model, tokenizer

if model_name == "Meta-Llama-2-7B-chat-hf":
    llama_model, tokenizer = get_model_and_tokenizer('../Meta-Llama-2-7B-chat-hf')


def call_llama_model(prompt):
    global counter
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(llama_model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    preds = llama_model.generate(**inputs, max_new_tokens=300)
    decoded_preds = [tokenizer.decode(pred[input_length:], skip_special_tokens=True) for pred in preds]
    counter += 1
    return decoded_preds[0]

def call_api(prompt):
    global counter
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ])
    counter += 1
    return completion.choices[0].message.content


from tqdm import tqdm
import time

prompt_template = task_to_prompts[TASK] if model_name != "Meta-Llama-2-7B-chat-hf" else llama_task_to_prompts[TASK]
for index, row in tqdm(df.iterrows()):
    if index < begin_index:
        continue
    profiles = row['profile']
    for profile in tqdm(profiles):
        if (
            'non_personalized_response' not in profile 
            or not profile['non_personalized_response'] 
            or model_name not in profile['non_personalized_response']
            or not profile['non_personalized_response'][model_name]
        ):
            if TASK == 'topic':
                prompt = prompt_template.format(profile['summary'])
            elif TASK == 'review':
                prompt = prompt_template.format(profile['overall'], profile['description'], profile['summary'])

            if model_name == "Meta-Llama-2-7B-chat-hf":
                non_personalized_response = call_llama_model(prompt)
            else:
                non_personalized_response = call_api(prompt)

            if 'non_personalized_response' not in profile or not profile['non_personalized_response']:
                profile['non_personalized_response'] = {model_name: non_personalized_response}
            else:
                profile['non_personalized_response'][model_name] = non_personalized_response
            # time.sleep(0.1)
    if index % 1 == 0:
        df.to_parquet(data_name + f'-nonPers_{begin_index}_{end_index}.parquet')
        print('Saved progress at index:', index, 'total calls:', counter)
    if index >= end_index:
        exit()