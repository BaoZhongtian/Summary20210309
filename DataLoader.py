import os
import numpy
import pickle
import torch
import torch.utils.data as torch_utils_data
from torch.nn.functional import embedding


class CollateSummarization:
    def __call__(self, batch):
        xs = [v[0] for v in batch]
        ys = [v[1] for v in batch]

        return xs, ys


class SummarizationDataset(torch_utils_data.Dataset):
    def __init__(self, article, abstract):
        self.article, self.abstract = article, abstract

    def __len__(self):
        return len(self.article)

    def __getitem__(self, index):
        return self.article[index], self.abstract[index]


def load_summarization(batch_size=8):
    load_path = 'Pretreatment'
    dictionary_embedding_raw = pickle.load(open(os.path.join(load_path, 'Dictionary_Embedding.pkl'), 'rb'))
    dictionary_embedding = []
    # print(dictionary_embedding_raw.keys())
    for index in range(numpy.max([sample for sample in dictionary_embedding_raw.keys()]) + 100):
        if index in dictionary_embedding_raw.keys():
            dictionary_embedding.append(dictionary_embedding_raw[index])
        else:
            dictionary_embedding.append(dictionary_embedding_raw[0])

    train_article = pickle.load(open(os.path.join(load_path, 'train-article.pkl'), 'rb'))
    train_abstract = pickle.load(open(os.path.join(load_path, 'train-abstract.pkl'), 'rb'))
    val_article = pickle.load(open(os.path.join(load_path, 'val-article.pkl'), 'rb'))
    val_abstract = pickle.load(open(os.path.join(load_path, 'val-abstract.pkl'), 'rb'))
    test_article = pickle.load(open(os.path.join(load_path, 'test-article.pkl'), 'rb'))
    test_abstract = pickle.load(open(os.path.join(load_path, 'test-abstract.pkl'), 'rb'))

    print(numpy.shape(dictionary_embedding))
    print(numpy.shape(train_article), numpy.shape(train_abstract), numpy.shape(val_article), numpy.shape(val_abstract),
          numpy.shape(test_article), numpy.shape(test_abstract))
    assert len(train_article) == len(train_abstract)
    assert len(val_article) == len(val_abstract)
    assert len(test_article) == len(test_abstract)

    train_dataset = SummarizationDataset(article=train_article, abstract=train_abstract)
    val_dataset = SummarizationDataset(article=val_article, abstract=val_abstract)
    test_dataset = SummarizationDataset(article=test_article, abstract=test_abstract)

    train_loader = torch_utils_data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateSummarization())
    val_loader = torch_utils_data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateSummarization())
    test_loader = torch_utils_data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateSummarization())
    return train_loader, val_loader, test_loader, dictionary_embedding


if __name__ == '__main__':
    train_loader, _, _, _ = load_summarization()
    for batch_index, [batch_article, batch_abstract] in enumerate(train_loader):
        print(batch_article)
        print(batch_abstract)
        exit()
