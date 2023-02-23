# coding=utf-8
import logging
import torch

import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from summarization_dataset import build_dataset


########################################################################
#
########################################################################

batch_size = 16
device = "cpu"

model_names = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
]

cuda_available = torch.cuda.is_available()


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# tokenized the dataset
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["prompt"])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


tokenizer = AutoTokenizer.from_pretrained(model_names[0])
tokenizer.pad_token = tokenizer.eos_token

_, _, eval_prompts, _ = build_dataset(
    tokenizer,
    batch_size,
    max_train_examples=1,
    max_eval_examples=64,
)

logging.warning("Tokenizing eval set...")
eval_dataset = Dataset.from_dict(
    {"prompt": eval_prompts}
)
eval_dataset = eval_dataset.map(tokenize, batched=False)
eval_dataset.set_format(type="torch")

dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=False,
)

# Arguments to pass to the `generate` function.
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

# ---------------------------------------------------
# generate summaries

model_outputs_dict = {}
logging.warning(f"Evaluating Models: {model_names}")

for model_name in model_names:
    logging.warning(f"Loading {model_name}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    model.eval()
    if cuda_available and device.startswith("cuda"):
        model.half()
        model.to(device)

    model_outputs_dict[model_name] = []

    logging.warning(f"summarization inferences...")
    for epoch, batch in tqdm(enumerate(dataloader)):
        query_tensors = batch["input_ids"]

        # generate text with the current model
        response_tensors = []
        for query in query_tensors:

            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len

            response = model.generate(
                torch.tensor(query).unsqueeze(dim=0).to(device),
                **generation_kwargs
            )
            response_tensors.append(response.squeeze()[-gen_len:])

        batch_response = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        model_outputs_dict[model_name].extend(batch_response)


# ---------------------------------------------------
# create csv file
csv_data = {"prompt": eval_prompts}
csv_data.update(model_outputs_dict)
df = pd.DataFrame(csv_data)
df.to_csv("./compare_trained_models.csv", sep=',', index=False)
