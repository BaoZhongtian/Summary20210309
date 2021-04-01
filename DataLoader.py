import os
import json
import tqdm
import numpy
import pickle

import torch
import pytorch_pretrained_bert
import torch.utils.data as torch_utils_data


class DatasetVae(torch_utils_data.Dataset):
    def __init__(self, sentence):
        self.sentence = sentence

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        return self.sentence[index]


class CollateVae:
    def __call__(self, batch):
        return batch


def loader_cnn_dm(batch_size=8, sentence_vae_flag=False):
    def load_json(data_path):
        data = []
        with open(data_path) as f:
            for line in tqdm.tqdm(f):
                data.append(json.loads(line))
        return data

    load_path = 'C:/PythonProject/MatchSum-master/MatchSum-master/data/'
    if sentence_vae_flag and os.path.exists('data/Test-IDs.pkl'):
        train_data = pickle.load(open('data/Train-IDs.pkl', 'rb'))
        val_data = pickle.load(open('data/Val-IDs.pkl', 'rb'))
        test_data = pickle.load(open('data/Test-IDs.pkl', 'rb'))
        print(len(train_data), len(val_data), len(test_data))

        train_dataset = DatasetVae(train_data)
        val_dataset = DatasetVae(val_data)
        test_dataset = DatasetVae(test_data)
        train_loader = torch_utils_data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateVae())
        val_loader = torch_utils_data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateVae())
        test_loader = torch_utils_data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateVae())
        return train_loader, val_loader, test_loader

    train_data = load_json(load_path + 'train_CNNDM_bert.jsonl')
    val_data = load_json(load_path + 'val_CNNDM_bert.jsonl')
    test_data = load_json(load_path + 'test_CNNDM_bert.jsonl')
    print(len(train_data), len(val_data), len(test_data))

    if sentence_vae_flag:
        tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-base-uncased')
        sentence_treatment(train_data, tokenizer, save_name='Train')
        sentence_treatment(val_data, tokenizer, save_name='Val')
        sentence_treatment(test_data, tokenizer, save_name='Test')
        loader_cnn_dm(batch_size=batch_size, sentence_vae_flag=sentence_vae_flag)


def sentence_treatment(input_data, tokenizer, save_name):
    total_ids = []
    if not os.path.exists('data'): os.makedirs('data')

    for index in tqdm.trange(len(input_data)):
        sample_text = input_data[index]['text']

        for sample in sample_text:
            total_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample)))
    pickle.dump(total_ids, file=open('data/%s-IDs.pkl' % save_name, 'wb'))


if __name__ == '__main__':
    train_loader, val_loader, test_loader = loader_cnn_dm(sentence_vae_flag=True)
    for batch_index, batch_data in enumerate(test_loader):
        print(batch_index, batch_data)
        for sample in batch_data:
            print(sample)
        exit()
