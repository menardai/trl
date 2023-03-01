import evaluate
import random
import torch

import numpy as np

from sft_tldr_dataset import TLDRDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


model_name = "gpt2-large"
output_dir = f"{model_name}-sft-tldr-checkpoint"

num_train_epochs = 4
train_batch_size = 16
gradient_accumulation_steps = 1

learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 1000
max_input_length = 550
save_steps = 1000
random.seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id

cuda_available = torch.cuda.is_available()

# Set up the datasets
data_path = "CarperAI/openai_summarize_tldr"

train_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "train",
    max_length=max_input_length,
)
eval_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "valid",
    max_length=max_input_length,
)

# Set up the metric
rouge = evaluate.load("rouge")


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result


# Create a preprocessing function to extract out the proper logits from the model output
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


training_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    gradient_accumulation_steps=gradient_accumulation_steps,

    output_dir=output_dir,
    save_total_limit=2,

    evaluation_strategy="steps",
    eval_accumulation_steps=1,

    fp16=cuda_available,
    half_precision_backend="auto",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_checkpointing=True,
    learning_rate=learning_rate,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=100,
    eval_steps=eval_steps,
    save_steps=save_steps,
    load_best_model_at_end=True,
    logging_steps=50,
    # deepspeed="./ds_config_gptj.json",
)

# Prepare the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
