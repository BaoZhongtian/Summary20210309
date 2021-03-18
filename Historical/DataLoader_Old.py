import numpy
import tqdm
import os
import pickle
import torch
import torch.utils.data as torch_utils_data
import pytorch_pretrained_bert

import Model


class CollateSummarization:
    def __call__(self, batch):
        xs = [v[0] for v in batch]
        ys = [v[1] for v in batch]
        zs = [v[2] for v in batch]
        return xs, ys, zs


class CollateTopicSummarization:
    def __call__(self, batch):
        ws = [v[0] for v in batch]
        xs = [v[1] for v in batch]
        ys = [v[2] for v in batch]
        zs = [v[3] for v in batch]
        return ws, xs, ys, zs


class CollateWordBag:
    def __init__(self, dictionary, paragraph_number):
        self.paragraph_number = paragraph_number
        self.index2dictionary, self.dictionary2index = {}, {}
        counter = 0
        for sample in dictionary:
            self.index2dictionary[counter] = sample
            self.dictionary2index[sample] = counter
            counter += 1

    def __call__(self, batch):
        xs = batch

        xs_return = []
        for indexX in range(len(xs)):
            part_length = numpy.int(numpy.ceil(len(xs[indexX]) / self.paragraph_number))
            sample = []
            for part_index in range(self.paragraph_number):
                word_bag_sample = numpy.zeros(len(self.index2dictionary.keys()))
                part_sample = xs[indexX][part_index * part_length:(part_index + 1) * part_length]
                for indexY in range(len(part_sample)):
                    word_bag_sample[self.dictionary2index[part_sample[indexY]]] += 1
                sample.append(word_bag_sample)
            xs_return.append(sample)

        return torch.FloatTensor(xs_return)


class SummarizationDataset(torch_utils_data.Dataset):
    def __init__(self, article, abstract, dictionary):
        self.article, self.abstract, self.dictionary = article, abstract, dictionary

    def __len__(self):
        return len(self.article)

    def __getitem__(self, index):
        article_tokens, abstract_tokens = [], []
        for sub_index in range(len(self.article[index])):
            article_tokens.append(self.dictionary[self.article[index][sub_index]])
        for sub_index in range(len(self.abstract[index])):
            abstract_tokens.append(self.dictionary[self.abstract[index][sub_index]])

        return torch.FloatTensor(numpy.array(article_tokens)), torch.FloatTensor(
            numpy.array(abstract_tokens)), self.abstract[index]


class SummarizationWordBagDataset(torch_utils_data.Dataset):
    def __init__(self, article, abstract):
        self.article = article

    def __len__(self):
        return len(self.article)

    def __getitem__(self, index):
        return self.article[index]


class SummarizationWithVAEDataset(torch_utils_data.Dataset):
    def __init__(self, article, abstract, dictionary, topic):
        self.article, self.abstract, self.dictionary, self.topic = article, abstract, dictionary, topic
        assert len(article) == len(abstract) == len(topic)

    def __len__(self):
        return len(self.article)

    def __getitem__(self, index):
        article_tokens, abstract_tokens = [], []
        for sub_index in range(len(self.article[index])):
            article_tokens.append(self.dictionary[self.article[index][sub_index]])
        for sub_index in range(len(self.abstract[index])):
            abstract_tokens.append(self.dictionary[self.abstract[index][sub_index]])

        return torch.FloatTensor(numpy.array(article_tokens)), torch.FloatTensor(
            numpy.array(abstract_tokens)), self.abstract[index], torch.FloatTensor(self.topic[index])


def loader_summarization_initial():
    def find_text(val_raw_data, search_index):
        distance = 2
        sample = val_raw_data[search_index + distance][2:-1]
        while sample == b'':
            distance += 1
            sample = val_raw_data[search_index + distance][2:-1]
        sample = str(sample).replace('<s>', '').replace('</s>', '').replace('s>', '')
        return sample[3:-1]

    def load_part(part_name):
        current_article, current_abstract = [], []
        with open(os.path.join(load_path, part_name), 'rb') as file:
            val_raw_data = file.readlines()

        for sample in val_raw_data[0:50]:
            print(sample)
        exit()
        for searchIndex in tqdm.trange(len(val_raw_data)):
            if val_raw_data[searchIndex][1:len('\x07article')] == b'article':
                current_article.append(find_text(val_raw_data, searchIndex))

            if val_raw_data[searchIndex][1:len('\x08abstract')] == b'abstract':
                current_abstract.append(find_text(val_raw_data, searchIndex))

        return current_article, current_abstract

    load_path = 'C:/ProjectData/finished_files/'

    # train_article, train_abstract = load_part(part_name='train.bin')
    # val_article, val_abstract = load_part(part_name='val.bin')
    test_article, test_abstract = load_part(part_name='test.bin')

    ##########################################
    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

    word_set = set()
    # for treat_name in ['test_abstract']:
    for treat_name in ['train_article', 'train_abstract', 'val_article', 'val_abstract', 'test_article',
                       'test_abstract']:
        print('Treating', treat_name, 'Part')
        treat_source = None
        if treat_name == 'train_article': treat_source = train_article
        if treat_name == 'train_abstract': treat_source = train_abstract
        if treat_name == 'val_article': treat_source = val_article
        if treat_name == 'val_abstract': treat_source = val_abstract
        if treat_name == 'test_article': treat_source = test_article
        if treat_name == 'test_abstract': treat_source = test_abstract
        assert treat_source is not None

        token_result = []
        for index in tqdm.trange(len(treat_source)):
            start_position = 0

            token_sample = []
            while start_position < len(treat_source[index]):
                text = treat_source[index][treat_source[index].find(' ', start_position):
                                           treat_source[index].find(' ', start_position + 500)]
                start_position += 500
                text = tokenizer.tokenize(text)
                text_id = tokenizer.convert_tokens_to_ids(text)
                token_sample.extend(text_id)

                for sample in text_id: word_set.add(sample)

            token_result.append(token_sample)
        if not os.path.exists('Data'):
            os.makedirs('Data')
        pickle.dump(token_result, open(os.path.join('Data', treat_name + '.pkl'), 'wb'))

    print('Total', len(word_set), 'words.')
    pickle.dump(word_set, open(os.path.join('Data', 'Dictionary.pkl'), 'wb'))
    print('Initial Treat Completed')


def loader_summarization(batch_size=32, cuda_flag=True, word_bag_flag=False, paragraph_number=None, topic_flag=False,
                         topic_name=None):
    load_path = 'C:/ProjectData/'
    if not os.path.exists(load_path + 'Data/Dictionary.pkl'): loader_summarization_initial()
    dictionary = pickle.load(open(load_path + 'Data/Dictionary.pkl', 'rb'))

    dictionary.add(101)
    dictionary.add(102)
    if not os.path.exists(load_path + 'Data/Dictionary_Embedding.pkl'):
        print('Generate Embedding')
        dictionary_list, dictionary_embedding = [], {}
        for sample in dictionary: dictionary_list.append(sample)

        model = Model.BertModelRawForEmbedding.from_pretrained('bert-large-uncased')
        if cuda_flag: model.cuda()

        for batch_start_position in tqdm.trange(0, len(dictionary_list), 32):
            batch_data = dictionary_list[batch_start_position:batch_start_position + 32]
            batch_data_raw = batch_data.copy()
            batch_data = torch.tensor(batch_data).view(1, -1)
            if cuda_flag: batch_data = batch_data.cuda()
            word_embedding = model(batch_data)
            word_embedding = word_embedding.squeeze().cpu().detach().numpy()

            for index in range(len(batch_data_raw)):
                dictionary_embedding[batch_data_raw[index]] = numpy.array(word_embedding[index])
        pickle.dump(dictionary_embedding, open(load_path + 'Data/Dictionary_Embedding.pkl', 'wb'))
    dictionary_embedding = pickle.load(open(load_path + 'Data/Dictionary_Embedding.pkl', 'rb'))

    print('Reading pickle data...')
    train_article = pickle.load(open(load_path + 'Data/train_article.pkl', 'rb'))[0:10000]
    train_abstract = pickle.load(open(load_path + 'Data/train_abstract.pkl', 'rb'))[0:10000]
    val_article = pickle.load(open(load_path + 'Data/val_article.pkl', 'rb'))
    val_abstract = pickle.load(open(load_path + 'Data/val_abstract.pkl', 'rb'))
    test_article = pickle.load(open(load_path + 'Data/test_article.pkl', 'rb'))
    test_abstract = pickle.load(open(load_path + 'Data/test_abstract.pkl', 'rb'))

    if topic_flag:
        train_vae = numpy.load(load_path + 'Data_VAE/%s-Train.npy' % topic_name)[0:10000]
        val_vae = numpy.load(load_path + 'Data_VAE/%s-Val.npy' % topic_name)
        test_vae = numpy.load(load_path + 'Data_VAE/%s-Test.npy' % topic_name)
        print(numpy.shape(train_vae), numpy.shape(val_vae), numpy.shape(test_vae))

        train_dataset = SummarizationWithVAEDataset(article=train_article, abstract=train_abstract,
                                                    dictionary=dictionary_embedding, topic=train_vae)
        val_dataset = SummarizationWithVAEDataset(article=val_article, abstract=val_abstract,
                                                  dictionary=dictionary_embedding, topic=val_vae)
        test_dataset = SummarizationWithVAEDataset(article=test_article, abstract=test_abstract,
                                                   dictionary=dictionary_embedding, topic=test_vae)
        train_loader = torch_utils_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=CollateTopicSummarization())
        val_loader = torch_utils_data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                                 collate_fn=CollateTopicSummarization())
        test_loader = torch_utils_data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  collate_fn=CollateTopicSummarization())
        return train_loader, val_loader, test_loader, dictionary_embedding

    if not word_bag_flag:
        train_dataset = SummarizationDataset(article=train_article, abstract=train_abstract,
                                             dictionary=dictionary_embedding)
        val_dataset = SummarizationDataset(article=val_article, abstract=val_abstract,
                                           dictionary=dictionary_embedding)
        test_dataset = SummarizationDataset(article=test_article, abstract=test_abstract,
                                            dictionary=dictionary_embedding)
        train_loader = torch_utils_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=CollateSummarization())
        val_loader = torch_utils_data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                                 collate_fn=CollateSummarization())
        test_loader = torch_utils_data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  collate_fn=CollateSummarization())
    else:
        train_dataset = SummarizationWordBagDataset(article=train_article, abstract=train_abstract)
        val_dataset = SummarizationWordBagDataset(article=val_article, abstract=val_abstract)
        test_dataset = SummarizationWordBagDataset(article=test_article, abstract=test_abstract)
        train_loader = torch_utils_data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=CollateWordBag(dictionary=dictionary, paragraph_number=paragraph_number))
        val_loader = torch_utils_data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=CollateWordBag(dictionary=dictionary, paragraph_number=paragraph_number))
        test_loader = torch_utils_data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=CollateWordBag(dictionary=dictionary, paragraph_number=paragraph_number))

    return train_loader, val_loader, test_loader, dictionary_embedding


if __name__ == '__main__':
    loader_summarization_initial()
    exit()
    # print(str(b"abc", "utf - 8"))
    train_loader, val_loader, test_loader, dictionary_embedding = loader_summarization(
        batch_size=16, topic_flag=True, topic_name='Result-VAE-1')
    for batchIndex, [batchArticle, batchAbstract, batchAbstractLabel, batchTopic] in enumerate(test_loader):
        # print(batchIndex, numpy.shape(batchArticle), numpy.shape(batchAbstract), numpy.shape(batchAbstractLabel),
        #       numpy.shape(batchTopic))
        for index in range(len(batchArticle)):
            print(numpy.shape(batchArticle[index]), numpy.shape(batchAbstract[index]),
                  numpy.shape(batchAbstractLabel[index]), numpy.shape(batchTopic[index]))
        exit()
