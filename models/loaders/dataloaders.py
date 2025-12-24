from datasets import load_dataset
from torch.utils.data import DataLoader
import random

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def collate_fn(batch):
    questions = [item["question"] for item in batch]
    positives = [
        item["positive_ctxs"][
            random.randint(0, len(item["positive_ctxs"]) - 1)
        ]["text"]
        for item in batch
    ]

    q_inputs = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    p_inputs = tokenizer(
        positives,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    return {
        "q_inputs": q_inputs,   # dict(input_ids, attention_mask)
        "p_inputs": p_inputs
    }


def get_loader(file, seed, batch_size, buffer_size, shuffle, num_workers):
    dataset = load_dataset(
        'json', data_files=file, split='train', streaming=True)

    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=num_workers) #pin_memory=True)
    return dataloader
