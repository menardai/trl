import logging
import torch

from tqdm import tqdm

from datasets import Dataset, load_dataset


def get_prompt_dataset(tokenizer, prompts, max_length, ending_with_tldr_exclusively=True):
    """
    Get the prompt after decoding to make sure dictionary of prompts and summaries is consistent decode prompt
    """
    skip_enc_dec_count = 0
    skip_too_long_count = 0
    formatted_prompts = []
    for i in tqdm(range(len(prompts))):
        tmp_a = tokenizer.decode(
            tokenizer(
                prompts[i].split("TL;DR:")[0],
                truncation=True,
                max_length=max_length - 5,  # to make sure "TL;DR" don't get truncated
            )["input_ids"],
            skip_special_tokens=True,
        ).strip()

        too_long = False
        if ending_with_tldr_exclusively:
            prompt_before_tldr = prompts[i].split("TL;DR:")[0]
            # not too long if the truncated tokenized version still end with the same text
            too_long = prompt_before_tldr.rstrip()[-20:] != tmp_a.rstrip()[-20:]

        if not too_long:
            tmp = tmp_a + "\nTL;DR:"
            tmp_encoded = tokenizer(tmp, truncation=True, max_length=max_length)["input_ids"]
            tmp_decoded = tokenizer.decode(
                tmp_encoded,
                skip_special_tokens=True,
            ).strip()

            if tmp == tmp_decoded:
                formatted_prompts.append(tmp_decoded)
            else:
                skip_enc_dec_count += 1
        else:
            skip_too_long_count += 1

    if skip_enc_dec_count:
        logging.warning(f"--- Skipped {skip_enc_dec_count} entries (problem with token encoding/decoding) ---")
    if skip_too_long_count:
        logging.warning(f"--- Skipped {skip_too_long_count} entries (too long) ---")

    return formatted_prompts


def build_dataset(tokenizer, ppo_batch_size, max_train_examples=None, max_eval_examples=None, ending_with_tldr_exclusively=True):
    tokenizer.padding_side = "left"

    max_length_input = (
        # config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
        550 - 50
    )

    # Browse dataset content here:
    # https://huggingface.co/datasets/CarperAI/openai_summarize_tldr
    #
    # DatasetDict({
    #     train: Dataset({
    #         features: ['prompt', 'label'],
    #         num_rows: 116722
    #     })
    #     valid: Dataset({
    #         features: ['prompt', 'label'],
    #         num_rows: 6447
    #     })
    #     test: Dataset({
    #         features: ['prompt', 'label'],
    #         num_rows: 6553
    #     })
    # })
    dataset = load_dataset("CarperAI/openai_summarize_tldr")

    # Store data into prompt and label pairs
    # val_set {list: 6447}:
    #   [
    #     (
    #       "SUBREDDIT: r/AskReddit\nTITLE: How do you get someone out of your head?\nPOST: Hi,\nI'm 22, and I have been with my girlfriend for 5 years now. We recently moved together. We've always loved each other intensely.\n\nProblem, I recently started to have feelings for an other person (a friend). This person has had a boyfriend for now 3 years, and has absolutely no ideas. Those feelings were so strong, it was hard to hide them. After 2 months of me being distant and really sad, my girlfriend forced me to say what was bothering me. I'm not a good liar, and now she knows.\n\nWe decided to give us a week alone, I went to my parents. \n\nNow, I'm completely lost. I keep on thinking about this person, and I hate that. I would like for those feelings to go away, to leave me alone. But I can't.  \n\nWhat do I do? It's been 3 months now, and I'm just desperate.\nTL;DR: ",
    #       "long relationship; fell in love with an other person; admitted it; would like it to disappear, though it doesn't."
    #     ),
    #     (
    #       "SUBREDDIT: r/pettyrevenge\nTITLE: So, my mom woke me up with a loud TV.\nPOST: She was in her living room, watching TV. This was at about 8:30 in the morning, and she was exercising. She turned the TV up extra loud to hear it over her excercycle, and woke me up. I went in there asking for her to turn it down. She said she didn't have to; I explained that I always used headphones so she didn't have to deal with my noise and that she should give me a little more respect, given that I paid rent at the time.\n\nShe disagreed. I went back to my room, rather pissed off at the lack of equality. I had no lock on my door; but I had a dresser right next to it, so I pulled one of the drawers out enough so that it caused the door to not be openable. Then, I turned my speakers up really loud and blasted Gangnam Style on repeat, with the bass cranked up as high as it could go.\n\nIf you hate Gangnam Style for being overplayed, you will see why I chose that particular song. I personally don't mind it. But here's the thing about my bass; it vibrates the walls, making one hell of a lot of noise. Needless to say, my mom was not pleased and shut off the internet. But it was oh so worth it.\nTL;DR: ",
    #       "Mom had the TV on loud and woke me up, didn't care that I'd respected audio levels in the house, so I countered with playing Gangnam Style on repeat with the bass thumping through the walls."
    #     ),
    #     ...
    #   ]
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

    # Split contents into summaries and labels
    #
    # val_posts {tuple: 6447}:
    #   (
    #     "SUBREDDIT: r/AskReddit\nTITLE: How do you get someone out of your head?\nPOST: Hi,\nI'm 22, and I have been ... just desperate.\nTL;DR: ",
    #     "SUBREDDIT: r/pettyrevenge\nTITLE: So, my mom woke me up with a loud TV.\nPOST: She was in her living room, ... so worth it.\nTL;DR: ",
    #     ...
    #   )
    #
    # val_summaries {tuple: 6447}:
    #   (
    #     "long relationship; fell in love with an other person; admitted it; would like it to disappear, though it doesn't."
    #     "Mom had the TV on loud and woke me up, didn't care that I'd respected audio levels in the house, so I countered with playing Gangnam Style on repeat with the bass thumping through the walls."
    #     ...
    #   )
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries.
    #   - val_prompts:
    #     a copy of val_posts clipped to max_length_input with a guarantee to end with "\nTL;DR:"
    #
    #   - prompt_summary_dict (contains both train and val prompts):
    #     {
    #       "SUBREDDIT: r/AskReddit\nTITLE: How do you get someone out of your head?\nPOST: Hi,\nI'm 22, and I have been ... just desperate.\nTL;DR:": "long relationship; fell in love with an other person; admitted it; would like it to disappear, though it doesn't.",
    #       "SUBREDDIT: r/pettyrevenge\nTITLE: So, my mom woke me up with a loud TV.\nPOST: She was in her living room, ... so worth it.\nTL;DR:": "Mom had the TV on loud and woke me up, didn't care that I'd respected audio levels in the house, so I countered with playing Gangnam Style on repeat with the bass thumping through the walls."
    #       ...
    #     }
    prompt_summary_dict = {}

    if max_train_examples:
        train_posts = train_posts[:max_train_examples]
    if max_eval_examples:
        val_posts = val_posts[:max_eval_examples]

    logging.warning("Formatting train prompts...")
    train_prompts = get_prompt_dataset(tokenizer, train_posts, max_length_input, ending_with_tldr_exclusively)
    for i in range(len(train_prompts)):
        prompt_summary_dict[train_prompts[i]] = train_summaries[i]

    logging.warning("Formatting eval prompts...")
    val_prompts = get_prompt_dataset(tokenizer, val_posts, max_length_input, ending_with_tldr_exclusively)
    for i in range(len(val_prompts)):
        prompt_summary_dict[val_prompts[i]] = val_summaries[i]

    # crop number of examples to a multiple of batch size
    train_prompts = train_prompts[:int(len(train_prompts) / ppo_batch_size) * ppo_batch_size]

    # tokenized the training dataset
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    logging.warning("Tokenizing training set...")
    train_dataset = Dataset.from_dict(
        {"prompt": train_prompts}
    )
    train_dataset = train_dataset.map(tokenize, batched=False)
    train_dataset.set_format(type='torch')

    logging.warning("Tokenizing eval set...")
    val_dataset = Dataset.from_dict(
        {"prompt": val_prompts}
    )
    val_dataset = val_dataset.map(tokenize, batched=False)
    val_dataset.set_format(type='torch')

    # validate that all prompts can be found in prompt_summary_dict using the 'query' field
    logging.warning("Validating training prompts tokenizations...")
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    for batch in dataloader:
        if prompt_summary_dict.get(batch['query'][0]) is None:
            logging.error(f"--- error --- (token count = {len(batch['input_ids'][0])}) \n{batch['query'][0]}")

    return train_dataset, val_dataset, train_prompts, val_prompts, prompt_summary_dict
