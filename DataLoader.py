import os
import tqdm
import pickle
import numpy
import torch.utils.data as torch_utils_data


class CollateSummarization:
    def __pad__(self, data):
        data_length = [len(sample) for sample in data]
        data_max_length = max(data_length)
        data_pad = []
        for sample in data:
            data_pad.append(numpy.concatenate([sample, numpy.zeros(data_max_length - len(sample))]))
        return data_pad, data_length

    def __call__(self, batch):
        xs = [v[0] for v in batch]
        xs_pad, xs_length = self.__pad__(xs)
        ys = [v[1] for v in batch]
        ys_pad, ys_length = self.__pad__(ys)
        return xs_pad, xs_length, ys_pad, ys_length


class CollateTopicSummarization(CollateSummarization):
    def __init__(self, paragraph_number):
        super(CollateTopicSummarization, self).__init__()
        self.paragraph_number = paragraph_number

    def __separate_paragraph__(self, data, paragraph_number):
        data_topic = []
        for sample in data:
            paragraph_sample = []
            for paragraph_index in range(paragraph_number):
                paragraph_word_bag = numpy.zeros(30611)
                for word_index in range(int(len(sample) / paragraph_number * paragraph_index),
                                        int(len(sample) / paragraph_number * (paragraph_index + 1))):
                    paragraph_word_bag[sample[word_index]] += 1
                paragraph_sample.append(paragraph_word_bag)
            data_topic.append(paragraph_sample)
        return data_topic

    def __call__(self, batch):
        xs = [v[0] for v in batch]
        xs_pad, xs_length = self.__pad__(xs)
        ys = [v[1] for v in batch]
        ys_pad, ys_length = self.__pad__(ys)
        zs = [v[2] for v in batch]
        paragraph = self.__separate_paragraph__(data=zs, paragraph_number=self.paragraph_number)
        return xs_pad, xs_length, ys_pad, ys_length, paragraph


class SummarizationDataset(torch_utils_data.Dataset):
    def __init__(self, article, abstract):
        self.article, self.abstract = article, abstract

    def __len__(self):
        return len(self.article)

    def __getitem__(self, index):
        return self.article[index], self.abstract[index]


class SummarizationTopicDataset(torch_utils_data.Dataset):
    def __init__(self, article_cut, abstract, article_raw):
        self.article_cut, self.abstract, self.article_raw = article_cut, abstract, article_raw
        assert len(self.article_cut) == len(self.abstract) == len(article_raw)

    def __len__(self):
        return len(self.article_cut)

    def __getitem__(self, index):
        return self.article_cut[index], self.abstract[index], self.article_raw[index]


def load_summarization(batch_size=8, max_length=600, paragraph_number=None):
    load_path = 'Pretreatment'
    dictionary_embedding_raw = pickle.load(open(os.path.join(load_path, 'Dictionary_Embedding.pkl'), 'rb'))
    dictionary_embedding = []
    # print(dictionary_embedding_raw.keys())
    for index in range(numpy.max([sample for sample in dictionary_embedding_raw.keys()]) + 100):
        if index in dictionary_embedding_raw.keys():
            dictionary_embedding.append(dictionary_embedding_raw[index])
        else:
            dictionary_embedding.append(dictionary_embedding_raw[0])

    train_article = pickle.load(open(os.path.join(load_path, 'train-article.pkl'), 'rb'))[0:100]
    train_abstract = pickle.load(open(os.path.join(load_path, 'train-abstract.pkl'), 'rb'))[0:100]
    val_article = pickle.load(open(os.path.join(load_path, 'val-article.pkl'), 'rb'))
    val_abstract = pickle.load(open(os.path.join(load_path, 'val-abstract.pkl'), 'rb'))
    test_article = pickle.load(open(os.path.join(load_path, 'test-article.pkl'), 'rb'))
    test_abstract = pickle.load(open(os.path.join(load_path, 'test-abstract.pkl'), 'rb'))

    #########################################

    train_article_neo, val_article_neo, test_article_neo = [], [], []
    train_abstract_neo = []
    for sample in train_article:
        if len(sample) > max_length:
            train_article_neo.append(sample[0:max_length])
        else:
            train_article_neo.append(sample)
    for sample in val_article:
        if len(sample) > max_length:
            val_article_neo.append(sample[0:max_length])
        else:
            val_article_neo.append(sample)
    for sample in test_article:
        if len(sample) > max_length:
            test_article_neo.append(sample[0:max_length])
        else:
            test_article_neo.append(sample)
    ##########################
    for sample in train_abstract:
        if len(sample) > max_length:
            train_abstract_neo.append(sample[0:max_length])
        else:
            train_abstract_neo.append(sample)
    train_abstract = train_abstract_neo
    ##########################
    train_article_cut, val_article_cut, test_article_cut = train_article_neo, val_article_neo, test_article_neo

    print(numpy.shape(dictionary_embedding))
    print(numpy.shape(train_article_cut), numpy.shape(train_abstract), numpy.shape(val_article_cut),
          numpy.shape(val_abstract), numpy.shape(test_article_cut), numpy.shape(test_abstract))
    assert len(train_article_cut) == len(train_abstract)
    assert len(val_article_cut) == len(val_abstract)
    assert len(test_article_cut) == len(test_abstract)

    if paragraph_number is None:
        train_dataset = SummarizationDataset(article=train_article_cut, abstract=train_abstract)
        val_dataset = SummarizationDataset(article=val_article_cut, abstract=val_abstract)
        test_dataset = SummarizationDataset(article=test_article_cut, abstract=test_abstract)
        train_loader = torch_utils_data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateSummarization())
        val_loader = torch_utils_data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateSummarization())
        test_loader = torch_utils_data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateSummarization())
        return train_loader, val_loader, test_loader, dictionary_embedding
    else:
        train_dataset = SummarizationTopicDataset(
            article_cut=train_article_cut, article_raw=train_article, abstract=train_abstract)
        val_dataset = SummarizationTopicDataset(
            article_cut=val_article_cut, article_raw=val_article, abstract=val_abstract)
        test_dataset = SummarizationTopicDataset(
            article_cut=test_article_cut, article_raw=test_article, abstract=test_abstract)

        train_loader = torch_utils_data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=CollateTopicSummarization(paragraph_number))
        val_loader = torch_utils_data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=CollateTopicSummarization(paragraph_number))
        test_loader = torch_utils_data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=CollateTopicSummarization(paragraph_number))
        return train_loader, val_loader, test_loader, dictionary_embedding


if __name__ == '__main__':
    import pytorch_pretrained_bert

    train_loader, _, test_loader, _ = load_summarization(paragraph_number=5)
    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-large-uncased')
    for batch_index, [batch_article, batch_article_length, batch_abstract, batch_abstract_length,
                      paragraph] in enumerate(test_loader):
        for sample in batch_abstract:
            print(sample)
            print(tokenizer.convert_ids_to_tokens(sample))

        exit()
