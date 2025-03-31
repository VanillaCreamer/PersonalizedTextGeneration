import pandas as pd
import sys
task_to_names = {
    "topic": "topic_writing_user/test-00000-of-00001-nonPers.parquet",
    "abs": "abstract_generation_user/test-00000-of-00002-nonPers.parquet",
    "review": "product_review_user/test-00000-of-00001-nonPers.parquet",
}

TASK = sys.argv[1]
df = pd.read_parquet(f'./dataset_longlamp/{task_to_names[TASK]}')
split = 'no_sampling'


import evaluate
import os


rouge = evaluate.load('metrics/rouge')
meteor = evaluate.load('metrics/meteor')
def calculate_scores(prediction: str, reference: str) -> dict:
    """
    Calculate ROUGE and METEOR scores between reference and prediction strings.

    Args:
        reference (str): The ground truth reference text.
        prediction (str): The predicted text to compare against the reference.

    Returns:
        dict: A dictionary containing ROUGE and METEOR scores.
    """
    # Compute ROUGE score
    rouge_result = rouge.compute(predictions=[prediction], references=[reference])

    # Compute METEOR score
    meteor_result = meteor.compute(predictions=[prediction], references=[reference])

    # Combine results
    results = {
        "rouge": rouge_result,
        "meteor_score": meteor_result["meteor"]
    }

    return results




import json
import os
from tqdm import tqdm
import sys


agg = 'mean'
min_token = 'all'
model_name = 'gpt-3.5-turbo'
generated_path = f"personalized_response_test/{TASK}/{split}/{agg}/{model_name}/{min_token}/"
saved_scored_path = f"personalized_response_test/{TASK}/{split}/{agg}/{model_name}/{min_token}-removed-scored/"
generated_ids = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(generated_path)]

for i in tqdm(generated_ids):
    responses = json.load(open(generated_path + f'personalized_responses_{i}.json'))
    gt = df.iloc[i]['output']
    for r in responses:
        r['input'] = df.iloc[i]['input']
        pred = r['preds'].split(r['input'])[-1]
        score = calculate_scores(pred, gt)
        r['removed_pred'] = pred
        r['scores'] = score

    with open(saved_scored_path + f'scored_user_{i}.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

