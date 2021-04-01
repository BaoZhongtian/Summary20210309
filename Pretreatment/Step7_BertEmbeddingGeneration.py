import tqdm
import torch
import pickle
import numpy
from HistoricalVHTM.Model import BertModelRawForEmbedding

if __name__ == '__main__':
    cuda_flag = True

    word_set = pickle.load(open('Step5_Dictionary.pkl', 'rb'))
    word_set.add(0)
    word_list = []
    dictionary_embedding = {}
    for word in word_set: word_list.append(word)

    model = BertModelRawForEmbedding.from_pretrained('bert-large-uncased')
    if cuda_flag: model.cuda()

    for batch_start_position in tqdm.trange(0, len(word_list), 32):
        batch_data = word_list[batch_start_position:batch_start_position + 32]
        batch_data_raw = batch_data.copy()
        batch_data = torch.LongTensor(batch_data).view(1, -1)
        if cuda_flag: batch_data = batch_data.cuda()
        word_embedding = model(batch_data)
        word_embedding = word_embedding.squeeze().cpu().detach().numpy()

        for index in range(len(batch_data_raw)):
            dictionary_embedding[batch_data_raw[index]] = numpy.array(word_embedding[index])
    pickle.dump(dictionary_embedding, open('Dictionary_Embedding.pkl', 'wb'))
