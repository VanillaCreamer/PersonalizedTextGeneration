import pandas as pd
import json
import numpy as np
import sys
import os



task_to_names = {
    "topic": "topic_writing_user",
    "abs": "abstract_generation_user",
    "review": "product_review_user"
}

TASK = sys.argv[1]
baseline_name = sys.argv[2]   # 'rag' # choose from ['rag', 'vanilla']
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
if TASK == 'abs':
    df1 = pd.read_parquet(f'./dataset_longlamp/{task_to_names[TASK]}/test-00000-of-00002-nonPers.parquet')
    df2 = pd.read_parquet(f'./dataset_longlamp/{task_to_names[TASK]}/test-00001-of-00002-nonPers.parquet')
    df = pd.concat([df1, df2], ignore_index=True)
else:
    df = pd.read_parquet(f'./dataset_longlamp/{task_to_names[TASK]}/test-00000-of-00001-nonPers.parquet')
df.head()
saved_path = f"./personalized_response_test_{baseline_name}/{TASK}/no_sampling/"
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
saved_file = saved_path + 'personalized_response_users.json'


from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    return model, tokenizer
model_name = './Meta-Llama-2-7B-chat-hf'
model, tokenizer = get_model_and_tokenizer(model_name)

def read_and_write_json(saved_file, outs):
    if os.path.exists(saved_file):
        with open(saved_file, 'r') as f:
            json_data = json.load(f)
        json_data.append(outs)
        with open(saved_file, 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    else:
        with open(saved_file, 'w') as f:
            json.dump([outs], f, ensure_ascii=False, indent=4)

if baseline_name in ['bm', 'ct']:
    ranked_docs = json.load(open(f"./dataset_longlamp/{task_to_names[TASK]}/{baseline_name}_test.json"))

if os.path.exists(saved_file):
    existed_users = [user['user_id'] for user in json.load(open(saved_file))]
else:
    existed_users = []

def get_first_k_words(text, k):
    words = text.split()
    return " ".join(words[:k])



from tqdm import tqdm
for user_id in tqdm(range(len(df))):
    if user_id in existed_users:
        continue
    data = df.iloc[user_id].to_dict()
    queries = data['input']
    gt = data['output']
    if baseline_name in ['bm', 'ct']:
        if TASK == 'topic':
            assert ranked_docs[user_id]['author'] == data['author']
            doc1 = ranked_docs[user_id]['profile'][0]
            doc2 = ranked_docs[user_id]['profile'][1]
            context_template = "'{}' is a summary for '{}'." 
            queries = context_template.format(doc1['summary'], doc1['content']) + context_template.format(doc2['summary'], doc2['content']) + "Following the given patterns, {}".format(queries)

        elif TASK == 'review':
            assert ranked_docs[user_id]['reviewerId'] == data['reviewerId']
            doc1 = ranked_docs[user_id]['profile'][0]
            doc2 = ranked_docs[user_id]['profile'][1]
            context_template = "{} is a rating for the product with description {}. {} is summary for {}."
            context = context_template.format(doc1['overall'], doc1['description'], doc1['summary'], doc1['reviewText']) + context_template.format(doc2['overall'], doc2['description'], doc2['summary'], doc2['reviewText'])
            queries =  context + "Following the given patterns, {}".format(queries)

        elif TASK == 'abs':
            assert ranked_docs[user_id]['name'] == data['name']
            doc1 = ranked_docs[user_id]['profile'][0]
            doc2 = ranked_docs[user_id]['profile'][1]
            context_template = "'{}' is the abstract for the title '{}'."
            context = context_template.format(get_first_k_words(doc1['abstract'], 750), doc1['title']) + context_template.format(get_first_k_words(doc2['abstract'], 750), doc2['title'])
            queries = context + "Use the above title and abstracts as context to understand the style and language of the user and {}".format(queries)




    inputs = tokenizer(queries, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    preds = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    decoded_preds = [tokenizer.decode(pred[input_length:], skip_special_tokens=True) for pred in preds]

    outs = {
        'user_id': user_id,
        'input': queries,
        'ground_truth': gt,
        'preds': decoded_preds[0]
    }
    read_and_write_json(saved_file, outs)
        
