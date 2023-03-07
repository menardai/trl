# coding=utf-8
# Based on trl/examples/sentiment/scripts/gpt2-sentiment.py and CarperAI/trlx summarization ppo training
import logging
import torch
import os

# import bitsandbytes as bnb

from tqdm import tqdm

tqdm.pandas()

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler

from summarization_dataset import build_dataset


########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# Fine-tunes a GPT2 model on the summarization dataset using PPO
# (proximal policy optimization) using 2 GPUs.
#
# First initialize accelerate configuration with `accelerate config`
# and select fp16
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.

rw_batch_size = 4

ppo_batch_size = 1  # ppo_epochs MUST BE 1 to use these batch size values
gradient_accumulation_steps = 96  # Total batch size = batch_size * gradient_accumulation_steps

eval_batch_size = 1

frozen_layers = 0.90

max_train_examples = None
max_eval_examples = 4
eval_step_frequency = 10

log_with = "wandb"  # "wandb" or None
save_step_interval = 1000

cuda_available = torch.cuda.is_available()
rw_device = "cuda" if cuda_available else "cpu"


# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
output_model_name = "ppo_gpt2_small_acc"
config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=ppo_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    ppo_epochs=1,   # ppo epoch is done a single batch size
    log_with=log_with,
)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def freeze_layers(model, frozen_layers=0.80):
    # Freeze the first 80% of the hidden layers of the model backbone
    logging.info(f"Freezing {frozen_layers} of model layers.")

    layers = model.pretrained_model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int((1.0 - frozen_layers) * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)


def save_ppo(trained_model_dir):
    os.makedirs(trained_model_dir, exist_ok=True)
    model.save_pretrained(trained_model_dir)
    tokenizer.save_pretrained(trained_model_dir)


tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# (only for this model)
tokenizer.pad_token = tokenizer.eos_token

train_dataset, eval_dataset, train_prompts, val_prompts, prompt_summary_dict = build_dataset(
    tokenizer,
    ppo_batch_size,
    max_train_examples=max_train_examples,
    max_eval_examples=max_eval_examples,
)

eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=eval_batch_size,
    collate_fn=collator,
    shuffle=False,
)

# Now let's build the model, the reference model, and the tokenizer.
logging.warning(f"Loading Model and Reference Model... {config.model_name}")
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

logging.warning(f"--> model layers {frozen_layers} frozen <--")
if frozen_layers > 0:
    freeze_layers(model, frozen_layers=frozen_layers)

# ref_model = create_reference_model(model, num_shared_layers=10)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model.eval()
if cuda_available:
    ref_model.half()

# logging.warning("--> optim.Adam8bit <--")  # --- output same token all the time ---
# optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model,
    tokenizer,
    # optimizer=optimizer,
    dataset=train_dataset,
    data_collator=collator,
)

# ---------------------------------------
# RM Model
# ---------------------------------------

logging.warning("Loading Reward Model...")
rw_model_name = "Tristan/gpt2_reward_summarization"

# Load the pre-trained reward model
rw_tokenizer = AutoTokenizer.from_pretrained(rw_model_name)
rw_model = AutoModelForSequenceClassification.from_pretrained(rw_model_name, num_labels=1)
rw_model.eval()
if rw_device.startswith("cuda"):
    rw_model.half()
rw_model.to(rw_device)

# Need to do this for gpt2, because it doesn't have an official pad token.
rw_tokenizer.pad_token = rw_tokenizer.eos_token
rw_model.config.pad_token_id = rw_tokenizer.eos_token_id


def get_scores(samples):
    """Compute scores for the given list of string."""
    scores_list = []
    for i in range(0, len(samples), rw_batch_size):
        sub_samples = samples[i: i + rw_batch_size]

        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=550,  # max_length=config.train.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(rw_device)
        attn_masks = encodings_dict["attention_mask"].to(rw_device)

        with torch.no_grad():
            sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)[0]
            sub_scores = sub_scores.view(input_ids.shape[0], )

        scores_list.append(sub_scores)

    scores = torch.cat(scores_list, dim=0)
    return scores


def reward_fn(samples):
    """Compute rewards for the given list of string."""
    original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
    original_samples = [text + prompt_summary_dict[text.strip()] for text in original_samples]

    original_scores = get_scores(original_samples)
    scores = get_scores(samples)

    norms_scores = scores - original_scores
    return norms_scores


# ---------------------------------------
# PPO Training
# ---------------------------------------
logging.warning("Starting PPO Training...")

# We then define the arguments to pass to the `generate` function.
# These arguments are passed to the `generate` function of the PPOTrainer,
# which is a wrapper around the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}
output_min_length = 8
output_max_length = 50
output_length_sampler = LengthSampler(output_min_length, output_max_length)

stats_logger_crashed = False
model_device = next(model.parameters()).device

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch['input_ids']

    # --- Get response from gpt2 ---
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])

    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # --- Compute rewards ---
    texts = [q + r for q, r in zip(batch['query'], batch['response'])]
    rewards = reward_fn(texts)
    reward_tensors = [torch.FloatTensor(reward.to("cpu").type(torch.FloatTensor)) for reward in rewards]

    # --- Run a PPO step ---
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

    if step % save_step_interval == 0 and step > 0:
        save_ppo(f"./{output_model_name}_checkpoints/{output_model_name}_{step}")

    # --- Evaluation set ---
    eval_query_response = None
    if step % eval_step_frequency == 0:
        eval_query_response = {
            "query": [],
            "response": [],
        }
        for eval_batch in eval_dataloader:
            eval_query_tensors = [tensor.to(model_device) for tensor in eval_batch['input_ids']]
            eval_response_tensors = []

            for query in eval_query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len

                eval_response = ppo_trainer.generate(query, **generation_kwargs)
                eval_response_tensors.append(eval_response.squeeze()[-gen_len:])

            eval_query_response["response"].extend([tokenizer.decode(r.squeeze()) for r in eval_response_tensors])
            eval_query_response["query"].extend(eval_batch["query"])

    try:
        ppo_trainer.log_stats(
            stats, batch, rewards,
            log_query_response=eval_query_response is not None,
            eval_batch=eval_query_response
        )
    except ValueError as e:
        if not stats_logger_crashed:
            logging.error("Error logging stats")
        stats_logger_crashed = True

save_ppo(f"./{output_model_name}")
