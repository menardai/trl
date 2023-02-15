# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import torch

from tqdm import tqdm

tqdm.pandas()

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl import PPOTrainer, PPOTrainer2GPU, PPOConfig, AutoModelForCausalLMWithValueHead
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

# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=1,
)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# (only for this model)
tokenizer.pad_token = tokenizer.eos_token

train_dataset, train_prompts, val_prompts, prompt_summary_dict = build_dataset(tokenizer)

# Now let's build the model, the reference model, and the tokenizer.
logging.warning("Loading Model and Reference Model...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
# ppo_trainer = PPOTrainer2GPU(
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model,
    tokenizer,
    dataset=train_dataset,
    data_collator=collator,
)

# ---------------------------------------
# RM Model
# ---------------------------------------

logging.warning("Loading Reward Model...")
rw_model_name = "Tristan/gpt2_reward_summarization"
rw_device = "cpu"

# Load the pre-trained reward model
rw_tokenizer = AutoTokenizer.from_pretrained(rw_model_name)
rw_model = AutoModelForSequenceClassification.from_pretrained(rw_model_name, num_labels=1)
rw_model.eval()
# rw_model.half()
rw_model.to(rw_device)

# Need to do this for gpt2, because it doesn't have an official pad token.
rw_tokenizer.pad_token = rw_tokenizer.eos_token
rw_model.config.pad_token_id = rw_tokenizer.eos_token_id


def get_scores(samples):
    """Compute scores for the given list of string."""
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]

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
            sub_scores = sub_scores.view(batch_size, )

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

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
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
    reward_tensors = [torch.FloatTensor(reward) for reward in rewards]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # --- Run a PPO step ---
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

    ppo_trainer.log_stats(stats, batch, rewards)
