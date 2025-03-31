# from transformers import pipeline, BitsAndBytesConfig
import argparse
import json
import os

import torch
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


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

"""
task_names = {
    "topic": "topic_writing_user", "author"
    "abs": "abstract_generation_user", "name"
    "review": "product_review_user", 'reviewerId'
    "news": "news_headline_user",
    "scholarly": scholarly_title_user,
    "tweet": tweet_paraphrase_user
    
}
"""
TASK_NAME = "news_headline_user"

parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--model_name', type=str, default='../models/meta-llama/Meta-Llama-2-7B-chat-hf')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=512)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default=TASK_NAME)
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--task_lora', type=str,
                    default='./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt')
parser.add_argument('--access_token', type=str, default=None)

args = parser.parse_args()
model_name = args.model_name
task_name = args.task_name
batch_size = args.batch_size
k = args.k
# max_step = args.max_step
cutoff_len = args.cut_off
add_eos_token = False
max_epoch = args.max_epoch

# # 4 bit quantization inference
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    # max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
)

# 8-bit quantization inference
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_quant_type="nf8",
#     bnb_8bit_compute_dtype=torch.float16,
#     bnb_8bit_use_double_quant=True,
#    max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

# 16-bit quantization inference
# bnb_config = BitsAndBytesConfig(
#     load_in_16bit=True,
#     bnb_16bit_quant_type="bf16",
#     bnb_16bit_compute_dtype=torch.bfloat16,
#     bnb_16bit_use_double_quant=True,
#     max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.eos_token = "</s>"
tokenizer.pad_token = '[PAD]'
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # local_files_only=False,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # , "k_proj", "out_proj"
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = transformers.TrainingArguments(
    output_dir='outputs/',
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    # save_steps=1e9,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    # max_steps=max_step,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
    logging_strategy='no',
    save_total_limit=2
)

# with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
#     test_data = json.load(f)
# test_data = load_dataset("../datasets/LongLaMP/", TASK_NAME, split='test')

if TASK_NAME == "topic_writing_user":
    test_data = load_dataset("../datasets/LongLaMP/", TASK_NAME, split='test')
    prompt_template = "Generate the content for a Reddit post {summary}.\nContent:"
    prompt_template_full = "Generate the content for a Reddit post {summary}.\nContent:{content}"
elif TASK_NAME == "abstract_generation_user":
    test_data = load_dataset("../datasets/LongLaMP/", TASK_NAME, split='test')
    prompt_template = "Generate and abstract for the title {title}.\nAbstract:"
    prompt_template_full = "Generate and abstract for the title {title}.\nAbstract:{abstract}"
elif TASK_NAME == "product_review_user":
    test_data = load_dataset("../datasets/LongLaMP/", TASK_NAME, split='test')
    prompt_template = "Generate the review text written by a reviewer who has given an overall rating of {overall} for a product with description {description}. The summary of the review text is {summary}.\nReview:"
    prompt_template_full = "Generate the review text written by a reviewer who has given an overall rating of {overall} for a product with description {description}. The summary of the review text is {summary}.\nReview:{reviewText}"
elif TASK_NAME == "news_headline_user":
    with open(f"../datasets/LaMP/LaMP_4/dev/val_bm.json", 'r') as f:
        test_data = json.load(f)
    prompt_template = "Generate a headline for the following article.\narticle: {text} headline:"
    prompt_template_full = "Generate a headline for the following article.\narticle: {text} headline: {title}"
elif TASK_NAME == "scholarly_title_user":
    with open(f"../datasets/LaMP/LaMP_5/dev/val_bm.json", 'r') as f:
        test_data = json.load(f)
    prompt_template = "Generate a title for the following abstract of a paper.\n abstract: {abstract} title:"
    prompt_template_full = "Generate a title for the following abstract of a paper.\n abstract: {abstract} title: {title}"
elif TASK_NAME == "tweet_paraphrase_user":
    with open(f"../datasets/LaMP/LaMP_7/dev/val_bm.json", 'r') as f:
        test_data = json.load(f)
    prompt_template = "tweet:"
    prompt_template_full = "tweet: {text}"
else:
    prompt_template = None
    prompt_template_full = None


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['prompt']
    tokenized_full_prompt = tokenize(full_prompt)
    # if not train_on_inputs:
    user_prompt = data_point['full_prompt']

    tokenized_user_prompt = tokenize(
        user_prompt, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt


# training
from datasets import Dataset

# model = PeftModel.from_pretrained(model=base_model, model_id=args.task_lora, is_trainable=False)
# base_model = model.merge_and_unload()
# print_trainable_parameters(model)

pred_all = []

for i in tqdm(range(len(test_data))):
    if i == 500:
        break
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    profile = test_data[i]['profile']
    train_data = []
    for idx, p in enumerate(test_data[i]['profile']):
        prompt = prompt_template.format(**p)
        full_prompt = prompt_template_full.format(**p)

        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

    # print(train_data)

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt)  # .shuffle()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # if i < 100:
    #     ckpt_name = "./outputs/ckpt/{}/OPPU_LoRA-{}.pth".format(TASK_NAME, i)
    #     lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k}
    #     torch.save(lora_state_dict, ckpt_name)

    model.eval()
    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

    # test inference
    uid = test_data[i]["id"]
    query = test_data[i]["input"]
    gt = test_data[i]["output"]
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt", padding=True, return_token_type_ids=False)
        input_length = inputs["input_ids"].shape[1]
        inputs = inputs.to(model.device)
        with torch.autocast(device_type="cuda"):
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=300
            )

        pred = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    out = {
            "uid": uid,
            "input": query,
            "ground_truth": gt,
            "preds": pred
        }
    # pred_all.append()
    saved_file = "outputs/{0}.json".format(TASK_NAME)
    read_and_write_json(saved_file, out)

# saved_file = "outputs/{0}.json".format(TASK_NAME)
# with open(saved_file, 'a+') as f:
#     json.dump(pred_all, f, ensure_ascii=False, indent=4)
