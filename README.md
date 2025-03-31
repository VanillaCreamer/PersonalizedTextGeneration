# Personalized Text Generation with Contrastive Activation Steering


## Dependencies

Using the following main dependencies:
- Python 3.10.15
- torch 2.5.1
- transformers 4.47.0



## Usage
### Data Preparation
- Download [LaMP](https://lamp-benchmark.github.io/) and [LongLaMP](https://longlamp-benchmark.github.io/)
- Generate non-personalized responses as: `python generate_non_personalized_response.py review Meta-Llama-2-7B-chat-hf`

### Generate

- Run inference as:  `python apply_steering_vector_review.py	`
- Evaluation: `python eval.py`

### Baselines

#### SFT/DPO

- Run stf.py or dpo.py in folder `./baselines`

#### RAG

- Ranking documents: `python rank_profiles.py --ranker=bm25/contriever`
- Generate: `python baseline.py ct/bm`

