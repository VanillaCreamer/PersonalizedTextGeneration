import pandas as pd
import json
import numpy as np
import torch
data_path = "./dataset_longlamp/product_review_user/test-00000-of-00001-nonPers.parquet"
df = pd.read_parquet(data_path)


import sys
import os


# user_id = int(sys.argv[1])
# model_name = sys.argv[2]
# agg_name = sys.argv[3]
# min_token = sys.argv[4]


use_input = False
user_id = 1
model_name = 'gpt-3.5-turbo' 
agg_name = "mean" 
min_token = '-1'


use_input = False
STARTING_LAYER = 10
STARTING_MULTIPLIER = -1
TASK = 'review'

print(user_id, model_name, agg_name, min_token)
if use_input:
    saved_path = "./personalized_response_test/%s/with_inputs/%s/%s/%s/" % (TASK, agg_name, model_name, min_token)
else:
    saved_path = "./personalized_response_test/%s/no_sampling/%s/%s/%s/" % (TASK, agg_name, model_name, min_token)

if not os.path.exists(saved_path):
    os.makedirs(saved_path)
saved_file = saved_path + 'personalized_response_user_%d.json' % user_id



if os.path.exists(saved_file):
    existed_data = json.load(open(saved_file, 'r'))
    if existed_data[-1]['multiplier'] == 1.0 and existed_data[-1]['layer'] == 31:
        print('User %d has been processed' % user_id)
        exit()
    else:
        print('User %d has been processed until multiplier %.1f and layer %d' % (user_id, existed_data[-1]['multiplier'], existed_data[-1]['layer']))
        starting_multiplier = round(existed_data[-1]['multiplier'] * 10)
        starting_layer = int(existed_data[-1]['layer'])
else:
    starting_multiplier = STARTING_MULTIPLIER
    starting_layer = STARTING_LAYER

contrastive_template = "##Input##: {} \n ##Output##: {} \n"
input_template = "Generate the review text written by a reviewer who has given an overall rating of {} for a product with description {}. The summary of the review text is {}."


data = df.iloc[user_id].to_dict()

if use_input:
    train_data = []
    for i, d in enumerate(data['profile']):
        inputs = input_template.format(d['overall'], d['description'], d['summary'])
        train_data.append((contrastive_template.format(inputs, d['reviewText']), contrastive_template.format(inputs, d['non_personalized_response'][model_name])))
else:
    train_data = [(d['reviewText'], d['non_personalized_response'][model_name]) for d in data['profile']]
attention_weights = None

from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    return model, tokenizer

model_name = 'Meta-Llama-2-7B-chat-hf'
model, tokenizer = get_model_and_tokenizer(model_name)
from steering_vectors import SteeringVector, train_steering_vector
from aggregators import pca_aggregator, logistic_aggregator, mean_aggregator, attention_aggregator, pos_aggregator
agg_funcs = {
    'mean': mean_aggregator(),
    'pca': pca_aggregator(),
    'lr': logistic_aggregator(),
    'att': attention_aggregator(attention_weights),
    'pos': pos_aggregator()
}
steering_vector: SteeringVector = train_steering_vector(
    model,
    tokenizer,
    train_data,
    move_to_cpu=True,
    read_token_index=-1,
    show_progress=True,
    aggregator=agg_funcs[agg_name]
)
queries = data['input']
gt = data['output']

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



inputs = tokenizer(queries, return_tensors='pt')
inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_len = inputs['input_ids'].shape[1]

try:
    min_token = int(min_token)
    slices = slice(min_token, None)
except:
    if min_token == 'all':
        slices = slice(0, input_len)
    elif 'in' in min_token:
        start_pos, end_pos = min_token.split('_')[1:]
        start_pos, end_pos = int(start_pos), int(end_pos)
        start_pos = input_len - 1 if start_pos < 0 else start_pos
        end_pos = input_len if end_pos < 0 else end_pos
        slices = slice(start_pos, end_pos)

print(slices, flush=True)

if starting_multiplier > STARTING_MULTIPLIER or starting_layer > STARTING_LAYER:
    multiplier = starting_multiplier * 0.1
    for layer in range(starting_layer + 1, len(steering_vector.layer_activations)):
        layer_steering_vector = SteeringVector(
                layer_activations={layer: steering_vector.layer_activations[layer]}, 
                layer_type=steering_vector.layer_type)
        with layer_steering_vector.apply(model, multiplier=multiplier, min_token_index=None, token_indices=slices):
            preds = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            outs = {
                'multiplier': multiplier,
                'layer': layer,
                'preds': preds[0]
            }
            read_and_write_json(saved_file, outs)


for multiplier in range(STARTING_MULTIPLIER + 1, 16, 1):
    multiplier *= 0.1
    for layer in range(STARTING_LAYER, len(steering_vector.layer_activations)):
        layer_steering_vector = SteeringVector(
                layer_activations={layer: steering_vector.layer_activations[layer]}, 
                layer_type=steering_vector.layer_type)
        with layer_steering_vector.apply(model, multiplier=multiplier, token_indices=slices, min_token_index=None):
            preds = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            outs = {
                'multiplier': multiplier,
                'layer': layer,
                'preds': preds[0]
            }
            read_and_write_json(saved_file, outs)
        
