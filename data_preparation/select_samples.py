# Explore the data in folder /home/manuel/Downloads/documents_20231117/extracted/2021_1/TXT_2021 and load the samples as a HF dataset

import datasets
import os

BASE_DIR = "/home/manuel/Downloads/documents_20231117/extracted/2021_1/TXT_2021"


# Clean documents
def filter_function(example) -> bool:
    # Remove all documents with less than 1000 characters
    # Remove all documents with less than 10 lines

    return len(example['text']) > 1000 and len(example['text'].split('\n')) > 10 and len(example['text']) < 100000


def load_samples_from_dir(dir_path: str = BASE_DIR):
    for file in os.listdir(dir_path)[:1000]:
        with open(os.path.join(dir_path, file), 'r') as f:
            text = f.read()
            yield {'text': text, 'file': file}


dataset = datasets.Dataset.from_generator(load_samples_from_dir)

print(dataset)

dataset = dataset.filter(filter_function)
print("Clean dataset length: ", len(dataset))


# separate all documents longer than 500 words in slightly-overlapping sliding windows of 500 words
def sliding_window(example):
    text = example['text'][0]
    file = example['file'][0]
    words = text.split(' ')
    windows = []
    for i in range(0, len(words), 450):
        windows.append(' '.join(words[i:i + 500]))
    return {'text': windows, 'file': [f"{file}_{i}" for i in range(len(windows))]}


dataset = dataset.map(sliding_window, batched=True, remove_columns=['text', 'file'], batch_size=1)
print("Sliding window dataset length: ", len(dataset))

# Save the dataset to disk in data/gallica
dataset.save_to_disk('./data/gallica_dirty')
