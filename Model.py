import numpy
import torch
import pytorch_pretrained_bert


#############################
# For Embedding Part
#############################
class BertEmbeddingsForEmbedding(torch.nn.Module):
    def __init__(self, config):
        super(BertEmbeddingsForEmbedding, self).__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = pytorch_pretrained_bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertModelRawForEmbedding(pytorch_pretrained_bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelRawForEmbedding, self).__init__(config)
        self.embeddings = BertEmbeddingsForEmbedding(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        return embedding_output

#############################

#############################


#############################
# For Seq2Seq Part
#############################

class Seq2SeqBasic(torch.nn.Module):
    def __init__(self, dictionary_embedding, cuda_flag=True):
        super(Seq2SeqBasic, self).__init__()
        self.Dictionary = dictionary_embedding
        self.CudaFlag = cuda_flag

        self.SOS_Embedding = torch.cat([torch.FloatTensor(self.Dictionary[101]), torch.zeros(1024)]).unsqueeze(
            0).unsqueeze(0)
        if cuda_flag: self.SOS_Embedding = self.SOS_Embedding.cuda()

        self.index2dictionary, self.dictionary2index = {}, {}
        for index, key in enumerate(self.Dictionary.keys()):
            self.index2dictionary[index] = key
            self.dictionary2index[key] = index

        self.BLSTM_Encoder_Layer = torch.nn.LSTM(
            input_size=1024, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.LSTM_Decoder_Layer = torch.nn.LSTM(
            input_size=2048, hidden_size=1024, num_layers=1, batch_first=True)
        self.AttentionWeight_Layer = torch.nn.Linear(in_features=2048, out_features=1)
        self.Predict_Decoder_Layer = torch.nn.Linear(in_features=2048, out_features=len(self.index2dictionary.keys()))

        self.LossFunction = torch.nn.CrossEntropyLoss()

    def forward(self, article, abstract=None, abstract_label_raw=None):
        abstract_label = []
        for sample in abstract_label_raw:
            abstract_label.append(self.dictionary2index[sample.numpy()[0]])
        abstract_label = torch.LongTensor(abstract_label)
        if self.CudaFlag:
            article = article.cuda()
            abstract_label = abstract_label.cuda()

        encoder_output, lstm_state = self.BLSTM_Encoder_Layer(article)
        encoder_output = encoder_output.squeeze()
        assert len(encoder_output.size()) == 2

        # 2 1 512
        # 1 1 512
        # 1 1 1024
        decoder_input = self.SOS_Embedding
        decoder_state = [torch.cat([lstm_state[0][0], lstm_state[0][1]], dim=-1).unsqueeze(0),
                         torch.cat([lstm_state[1][0], lstm_state[1][1]], dim=-1).unsqueeze(0)]

        if abstract is not None:
            if self.CudaFlag: abstract = abstract.cuda()
            abstract = abstract.squeeze()
            decoder_predict_probability = []

            random_choose = numpy.random.random()
            for _ in range(len(abstract)):
                decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)

                decoder_repeat = decoder_output.squeeze(0).repeat([encoder_output.size()[0], 1])
                decoder_output_concat = torch.cat([decoder_repeat, encoder_output], dim=-1)
                decoder_attention_weight = self.AttentionWeight_Layer(decoder_output_concat).softmax(dim=0)
                decoder_attention_padding = decoder_attention_weight.repeat([1, 2048])
                decoder_weighted_raw = decoder_output_concat * decoder_attention_padding
                decoder_weighted = decoder_weighted_raw.sum(dim=0).unsqueeze(0)

                decoder_predict = self.Predict_Decoder_Layer(decoder_weighted)
                decoder_predict_probability.append(decoder_predict)
                # print(numpy.shape(decoder_predict))
                decoder_predict_value = decoder_predict.argmax(dim=-1)

                if random_choose > 0.5:
                    decoder_input = torch.FloatTensor(
                        self.Dictionary[self.index2dictionary[decoder_predict_value.detach().cpu().numpy()[0]]])
                else:
                    decoder_input = abstract[_]

                decoder_input = decoder_input.unsqueeze(0).unsqueeze(0)
                if self.CudaFlag: decoder_input = decoder_input.cuda()
                decoder_input = torch.cat([decoder_input, decoder_weighted.view([2, 1, 1024])[1:]], dim=-1)

            decoder_predict_probability = torch.cat(decoder_predict_probability, dim=0)

            loss = self.LossFunction(input=decoder_predict_probability, target=abstract_label)
            return loss


# class Seq2SeqBasicBatch(Seq2SeqBasic):
#     def __init__(self, dictionary_embedding, cuda_flag=True):
#         super(Seq2SeqBasicBatch, self).__init__(dictionary_embedding=dictionary_embedding, cuda_flag=cuda_flag)
#
#     def input_padding(self, article):
#         article_padding = []
#         article_length = torch.LongTensor([len(sample) for sample in article])
#         article_max_length = numpy.max([len(sample) for sample in article])
#         for sample in article:
#             article_sample = torch.cat([sample, torch.zeros([article_max_length - sample.size()[0], sample.size()[1]])],
#                                        dim=0).unsqueeze(0)
#             article_padding.append(article_sample)
#         article_padding = torch.cat(article_padding, dim=0)
#         return article_padding, article_length
#
#     def attention_mask(self, length):
#         max_length = torch.max(length).numpy()
#         attention_map = []
#         for length_sample in length:
#             sample_attention_map = torch.cat(
#                 [torch.ones(length_sample), -1 * torch.ones(max_length - length_sample.numpy())])
#             attention_map.append(sample_attention_map.unsqueeze(0))
#         attention_map = torch.cat(attention_map, dim=0).unsqueeze(-1) * 9999
#         if self.CudaFlag: attention_map = attention_map.cuda()
#         return attention_map
#
#     def forward(self, article, abstract=None, abstract_label_raw=None):
#         article_padding, article_length = self.input_padding(article=article)
#         article_attention_map = self.attention_mask(length=article_length)
#
#         if self.CudaFlag: article_padding = article_padding.cuda()
#         encoder_output, lstm_state = self.BLSTM_Encoder_Layer(article_padding)
#
#         decoder_input = self.SOS_Embedding.repeat([encoder_output.size()[0], 1, 1])
#         decoder_state = [torch.cat([lstm_state[0][0], lstm_state[0][1]], dim=-1).unsqueeze(0),
#                          torch.cat([lstm_state[1][0], lstm_state[1][1]], dim=-1).unsqueeze(0)]
#
#         ##############################################
#
#         abstract_padding, abstract_length = self.input_padding(article=abstract)
#         decoder_predict_probability = []
#         print(numpy.shape(abstract_padding))
#
#         for _ in range(abstract_padding.size()[1]):
#             decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)
#             decoder_repeat = decoder_output.repeat([1, encoder_output.size()[1], 1])
#             decoder_output_concat = torch.cat([decoder_repeat, encoder_output], dim=-1)
#             decoder_attention_weight = torch.min(self.AttentionWeight_Layer(decoder_output_concat),
#                                                  article_attention_map).softmax(dim=1)
#             decoder_attention_padding = decoder_attention_weight.repeat([1, 1, 2048])
#             decoder_weighted_raw = decoder_output_concat * decoder_attention_padding
#             decoder_weighted = decoder_weighted_raw.sum(dim=1)
#
#             decoder_predict = self.Predict_Decoder_Layer(decoder_weighted)
#             decoder_predict_probability.append(decoder_predict)
#             # print(numpy.shape(decoder_predict))
#             decoder_predict_value = decoder_predict.argmax(dim=-1)
#
#             decoder_input = [torch.FloatTensor(
#                 self.Dictionary[self.index2dictionary[decoder_predict_value.detach().cpu().numpy()[index]]]).unsqueeze(
#                 0) for index in range(encoder_output.size()[0])]
#             decoder_input = torch.cat(decoder_input, dim=0)
#
#             decoder_weighted_choose = decoder_weighted.view([decoder_weighted.size()[0], 2, 1024]).permute([1, 0, 2])[1]
#             if self.CudaFlag:
#                 decoder_input = decoder_input.cuda()
#                 decoder_weighted_choose = decoder_weighted_choose.cuda()
#             decoder_input = torch.cat([decoder_input, decoder_weighted_choose], dim=-1).unsqueeze(1)
#
#         print(decoder_predict_probability[0].size())
#
#         # decoder_predict_probability = torch.cat(decoder_predict_probability, dim=0)
#         # print(numpy.shape(decoder_predict_probability))


if __name__ == '__main__':
    import os
    from DataLoader import loader_summarization

    cuda_flag = True

    save_path = 'Result/BasicBatch'
    if not os.path.exists(save_path): os.makedirs(save_path)
    train_dataset, val_dataset, test_dataset, dictionary_embedding = loader_summarization(batch_size=4)

    seq2seqBasic = Seq2SeqBasicBatch(dictionary_embedding, cuda_flag=cuda_flag)
    if cuda_flag: seq2seqBasic.cuda()
    optimizer = torch.optim.Adam(params=seq2seqBasic.parameters(), lr=5E-4)

    for episode_index in range(100):
        total_loss = 0.0
    with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
        for batchIndex, [batchArticle, batchAbstract, batchAbstractLabel] in enumerate(train_dataset):
            loss = seq2seqBasic(batchArticle, batchAbstract, batchAbstractLabel)
            exit()
            loss_value = loss.cpu().detach().numpy()
            total_loss += loss_value
            print('\rBatch %d Loss = %f' % (batchIndex, loss_value), end='')
            file.write(str(loss_value) + '\n')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print('\nEpisode %d Total Loss = %f' % (episode_index, total_loss))
            # torch.save(obj={'ModelStateDict': seq2seqBasic.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
            #            f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))
            # torch.save(obj=seq2seqBasic, f=os.path.join(save_path, 'Network-%04d.pkl' % episode_index))
