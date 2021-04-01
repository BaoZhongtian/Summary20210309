import numpy
import torch
import pytorch_pretrained_bert
from torch.nn.functional import embedding


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
# Seq2Seq Model
#############################


class Seq2SeqBasic(torch.nn.Module):
    def __init__(self, lstm_size=128, cuda_flag=True):
        super(Seq2SeqBasic, self).__init__()
        self.cuda_flag = cuda_flag
        self.lstm_size = lstm_size
        self.BLSTM_Encoder_Layer = torch.nn.LSTM(
            input_size=1024, hidden_size=lstm_size, num_layers=1, batch_first=True, bidirectional=True)
        self.LSTM_Decoder_Layer = torch.nn.LSTM(
            input_size=1024, hidden_size=lstm_size * 2, num_layers=1, batch_first=True)
        self.Predict_Decoder_Layer = torch.nn.Linear(in_features=lstm_size * 2, out_features=30611)
        self.LossFunction = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, word_embedding, article, article_length, abstract=None, abstract_length=None, decode_flag=False):
        article = torch.LongTensor(article)
        article_length = torch.LongTensor(article_length)
        if abstract is not None: abstract = torch.LongTensor(abstract)
        if abstract_length is not None: abstract_length = torch.LongTensor(abstract_length)
        input_word = torch.LongTensor(numpy.ones(article.size()[0]) * 101)

        if self.cuda_flag:
            article = article.cuda()
            word_embedding = word_embedding.cuda()
            input_word = input_word.cuda()

        article_embedding_result = embedding(input=article, weight=word_embedding)
        article_encoder_output, article_encoder_state = self.BLSTM_Encoder_Layer(article_embedding_result)

        decoder_input = embedding(input=input_word, weight=word_embedding).unsqueeze(1)
        decoder_state = [torch.cat([article_encoder_state[0][0], article_encoder_state[0][1]], dim=-1).unsqueeze(0),
                         torch.cat([article_encoder_state[1][0], article_encoder_state[1][1]], dim=-1).unsqueeze(0)]

        if not decode_flag:
            random_choose = numpy.random.random()

            if self.cuda_flag:
                decoder_input = decoder_input.cuda()
                abstract = abstract.cuda()

            decoder_predict_probability = []
            for word_index in range(abstract.size()[1]):
                decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)

                decoder_predict = self.Predict_Decoder_Layer(decoder_output)
                decoder_predict_probability.append(decoder_predict)
                # print(numpy.shape(decoder_predict))
                decoder_predict_value = decoder_predict.argmax(dim=-1)

                ecoder_input = embedding(input=decoder_predict_value, weight=word_embedding)
                # if random_choose < 0.5:
                #     decoder_input = embedding(input=decoder_predict_value, weight=word_embedding)
                # else:
                #     # print(word_index, numpy.shape(abstract))
                #     decoder_input = embedding(input=abstract[:, word_index:word_index + 1], weight=word_embedding)

            decoder_predict_probability = torch.cat(decoder_predict_probability, dim=1)

            decoder_predict_probability = decoder_predict_probability.view(
                [decoder_predict_probability.size()[0] * decoder_predict_probability.size()[1],
                 decoder_predict_probability.size()[2]])
            abstract = abstract.view([abstract.size()[0] * abstract.size()[1]])

            loss = self.LossFunction(input=decoder_predict_probability, target=abstract)

            return loss
        else:
            decoder_predict_probability = []
            for word_index in range(abstract.size()[1]):
                decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)
                decoder_predict = self.Predict_Decoder_Layer(decoder_output)
                decoder_predict_probability.append(decoder_predict)
                # print(numpy.shape(decoder_predict))
                decoder_predict_value = decoder_predict.argmax(dim=-1)
                decoder_input = embedding(input=decoder_predict_value, weight=word_embedding)

            decoder_predict_probability = torch.cat(decoder_predict_probability, dim=1)
            decoder_predict_result = decoder_predict_probability.argmax(dim=-1)
            return decoder_predict_result


class Seq2SeqWAttention(Seq2SeqBasic):
    def __init__(self, lstm_size=128, cuda_flag=True):
        super(Seq2SeqWAttention, self).__init__(lstm_size=lstm_size, cuda_flag=cuda_flag)
        self.LSTM_Decoder_Layer = torch.nn.LSTM(
            input_size=1024 + lstm_size * 2, hidden_size=lstm_size * 2, num_layers=1, batch_first=True)
        self.AttentionWeight_H_Layer = torch.nn.Linear(in_features=lstm_size * 2, out_features=64)
        self.AttentionWeight_S_Layer = torch.nn.Linear(in_features=lstm_size * 2, out_features=64)
        self.AttentionWeight_Final_Layer = torch.nn.Linear(in_features=64, out_features=1)

    def mask_generation(self, length):
        max_length = torch.max(length)

        attention_mask = []
        for index in range(len(length)):
            attention_mask.append(
                torch.cat([torch.ones(length[index]), torch.ones(max_length - length[index]) * -1]).unsqueeze(0))

        attention_mask = torch.cat(attention_mask, dim=0) * 9999
        return attention_mask

    def forward(self, word_embedding, article, article_length, abstract=None, abstract_length=None, decode_flag=False):
        article = torch.LongTensor(article)
        article_length = torch.LongTensor(article_length)
        article_mask = self.mask_generation(length=article_length)
        if abstract is not None: abstract = torch.LongTensor(abstract)
        if abstract_length is not None: abstract_length = torch.LongTensor(abstract_length)
        input_word = torch.LongTensor(numpy.ones(article.size()[0]) * 101)

        if self.cuda_flag:
            article = article.cuda()
            article_mask = article_mask.cuda()
            word_embedding = word_embedding.cuda()
            input_word = input_word.cuda()

        article_embedding_result = embedding(input=article, weight=word_embedding)
        article_encoder_output, article_encoder_state = self.BLSTM_Encoder_Layer(article_embedding_result)

        decoder_input = embedding(input=input_word, weight=word_embedding).unsqueeze(1)
        if self.cuda_flag:
            decoder_input = torch.cat(
                [decoder_input, torch.zeros([decoder_input.size()[0], 1, self.lstm_size * 2]).cuda()], dim=-1)
        else:
            decoder_input = torch.cat(
                [decoder_input, torch.zeros([decoder_input.size()[0], 1, self.lstm_size * 2])], dim=-1)
        decoder_state = [torch.cat([article_encoder_state[0][0], article_encoder_state[0][1]], dim=-1).unsqueeze(0),
                         torch.cat([article_encoder_state[1][0], article_encoder_state[1][1]], dim=-1).unsqueeze(0)]

        if not decode_flag:
            random_choose = numpy.random.random()

            if self.cuda_flag:
                decoder_input = decoder_input.cuda()
                abstract = abstract.cuda()

            decoder_predict_probability = []
            encoder_weight_s = self.AttentionWeight_S_Layer(article_encoder_output)
            for word_index in range(abstract.size()[1]):
                decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)
                decoder_predict = self.Predict_Decoder_Layer(decoder_output)
                decoder_predict_probability.append(decoder_predict)
                decoder_predict_value = decoder_predict.argmax(dim=-1)
                decoder_input = embedding(input=abstract[:, word_index:word_index + 1], weight=word_embedding)
                # if random_choose < 0.5:
                #     decoder_input = embedding(input=decoder_predict_value, weight=word_embedding)
                # else:
                #     decoder_input = embedding(input=abstract[:, word_index:word_index + 1], weight=word_embedding)

                decoder_weight_h = self.AttentionWeight_H_Layer(decoder_output).repeat([1, article.size()[1], 1])
                decoder_weight_add = encoder_weight_s + decoder_weight_h
                decoder_weight_add = decoder_weight_add.tanh()
                decoder_weight = self.AttentionWeight_Final_Layer(decoder_weight_add).squeeze()
                decoder_weight = torch.min(decoder_weight, article_mask).softmax(dim=-1)

                decoder_weight_pad = decoder_weight.unsqueeze(-1).repeat([1, 1, self.lstm_size * 2])
                decoder_attention_result = article_encoder_output * decoder_weight_pad
                decoder_attention_result = torch.sum(decoder_attention_result, dim=1).unsqueeze(1)
                decoder_input = torch.cat([decoder_input, decoder_attention_result], dim=-1)
                # print(numpy.shape(decoder_input), numpy.shape(decoder_attention_result))
                # exit()

            decoder_predict_probability = torch.cat(decoder_predict_probability, dim=1)

            decoder_predict_probability = decoder_predict_probability.view(
                [decoder_predict_probability.size()[0] * decoder_predict_probability.size()[1],
                 decoder_predict_probability.size()[2]])
            abstract = abstract.view([abstract.size()[0] * abstract.size()[1]])

            loss = self.LossFunction(input=decoder_predict_probability, target=abstract)

            return loss

        else:
            if self.cuda_flag:
                decoder_input = decoder_input.cuda()
                abstract = abstract.cuda()

            decoder_predict_probability = []
            encoder_weight_s = self.AttentionWeight_S_Layer(article_encoder_output)
            for word_index in range(abstract.size()[1]):
                decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)
                decoder_predict = self.Predict_Decoder_Layer(decoder_output)
                decoder_predict_probability.append(decoder_predict)
                decoder_predict_value = decoder_predict.argmax(dim=-1)
                decoder_input = embedding(input=decoder_predict_value, weight=word_embedding)

                decoder_weight_h = self.AttentionWeight_H_Layer(decoder_output).repeat([1, article.size()[1], 1])
                decoder_weight_add = encoder_weight_s + decoder_weight_h
                decoder_weight_add = decoder_weight_add.tanh()
                decoder_weight = self.AttentionWeight_Final_Layer(decoder_weight_add).squeeze()
                decoder_weight = torch.min(decoder_weight, article_mask).softmax(dim=-1)

                decoder_weight_pad = decoder_weight.unsqueeze(-1).repeat([1, 1, self.lstm_size * 2])
                decoder_attention_result = article_encoder_output * decoder_weight_pad
                decoder_attention_result = torch.sum(decoder_attention_result, dim=1).unsqueeze(1)
                decoder_input = torch.cat([decoder_input, decoder_attention_result], dim=-1)
                # print(numpy.shape(decoder_input), numpy.shape(decoder_attention_result))
                # exit()

            decoder_predict_probability = torch.cat(decoder_predict_probability, dim=1)
            decoder_predict_result = decoder_predict_probability.argmax(dim=-1)
            return decoder_predict_result



class Seq2SeqWTopic(Seq2SeqWAttention):
    def __init__(self, lstm_size=128, cuda_flag=True):
        super(Seq2SeqWTopic, self).__init__(lstm_size=lstm_size, cuda_flag=cuda_flag)
        self.LSTM_Decoder_Layer = torch.nn.LSTM(
            input_size=2048, hidden_size=lstm_size * 2, num_layers=1, batch_first=True)

        self.Topic_Weight_Layer1st = torch.nn.Linear(in_features=30611, out_features=1024)
        self.Topic_Weight_Layer2nd = torch.nn.Linear(in_features=1024, out_features=1024)
        self.Topic_Average_Layer = torch.nn.Linear(in_features=1024, out_features=768)
        self.Topic_Std_Layer = torch.nn.Linear(in_features=1024, out_features=768)

        self.TopicWeight_W_Layer = torch.nn.Linear(in_features=768, out_features=64)
        self.TopicWeight_S_Layer = torch.nn.Linear(in_features=lstm_size * 2, out_features=64)
        self.TopicWeight_Final_Layer = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, word_embedding, article, article_length, abstract=None, abstract_length=None,
                article_paragraph=None, decode_flag=False):
        article = torch.LongTensor(article)
        article_length = torch.LongTensor(article_length)
        article_mask = self.mask_generation(length=article_length)
        if abstract is not None: abstract = torch.LongTensor(abstract)
        if abstract_length is not None: abstract_length = torch.LongTensor(abstract_length)
        input_word = torch.LongTensor(numpy.ones(article.size()[0]) * 101)

        assert article_paragraph is not None
        article_paragraph = torch.FloatTensor(article_paragraph)

        if self.cuda_flag:
            article = article.cuda()
            article_mask = article_mask.cuda()
            word_embedding = word_embedding.cuda()
            input_word = input_word.cuda()
            article_paragraph = article_paragraph.cuda()

        article_embedding_result = embedding(input=article, weight=word_embedding)
        article_encoder_output, article_encoder_state = self.BLSTM_Encoder_Layer(article_embedding_result)

        decoder_input = embedding(input=input_word, weight=word_embedding).unsqueeze(1)
        if self.cuda_flag:
            decoder_input = torch.cat(
                [decoder_input, torch.zeros([decoder_input.size()[0], 1, 1024]).cuda()], dim=-1)
        else:
            decoder_input = torch.cat(
                [decoder_input, torch.zeros([decoder_input.size()[0], 1, 1024])], dim=-1)
        # print(numpy.shape(decoder_input))

        decoder_state = [torch.cat([article_encoder_state[0][0], article_encoder_state[0][1]], dim=-1).unsqueeze(0),
                         torch.cat([article_encoder_state[1][0], article_encoder_state[1][1]], dim=-1).unsqueeze(0)]
        #######################################

        paragraph_1st = self.Topic_Weight_Layer1st(article_paragraph)
        paragraph_2nd = self.Topic_Weight_Layer2nd(paragraph_1st)
        paragraph_mean = self.Topic_Average_Layer(paragraph_2nd)
        paragraph_std = self.Topic_Std_Layer(paragraph_2nd)

        paragraph_vector = paragraph_mean + paragraph_std * torch.randn_like(paragraph_std)
        paragraph_weight_w = self.TopicWeight_W_Layer(paragraph_vector)

        #######################################

        if abstract is not None:
            random_choose = numpy.random.random()

            if self.cuda_flag:
                decoder_input = decoder_input.cuda()
                abstract = abstract.cuda()

            decoder_predict_probability = []
            encoder_weight_s = self.AttentionWeight_S_Layer(article_encoder_output)
            for word_index in range(abstract.size()[1]):
                decoder_output, decoder_state = self.LSTM_Decoder_Layer(decoder_input, hx=decoder_state)
                decoder_predict = self.Predict_Decoder_Layer(decoder_output)
                decoder_predict_probability.append(decoder_predict)
                decoder_predict_value = decoder_predict.argmax(dim=-1)

                decoder_input = embedding(input=abstract[:, word_index:word_index + 1], weight=word_embedding)
                # if random_choose < 0.5:
                #     decoder_input = embedding(input=decoder_predict_value, weight=word_embedding)
                # else:
                #     decoder_input = embedding(input=abstract[:, word_index:word_index + 1], weight=word_embedding)

                paragraph_weight_s = self.TopicWeight_S_Layer(decoder_output).repeat(
                    [1, paragraph_weight_w.size()[1], 1])
                paragraph_weight_add = paragraph_weight_w + paragraph_weight_s
                paragraph_weight_final = self.TopicWeight_Final_Layer(paragraph_weight_add)
                paragraph_weight_final = paragraph_weight_final.softmax(dim=1).repeat([1, 1, 768])

                topic_weighted_raw = paragraph_weight_final * paragraph_vector
                topic_weighted = topic_weighted_raw.sum(dim=1).unsqueeze(1)

                #####################################

                decoder_weight_h = self.AttentionWeight_H_Layer(decoder_output).repeat([1, article.size()[1], 1])
                decoder_weight_add = encoder_weight_s + decoder_weight_h
                decoder_weight_add = decoder_weight_add.tanh()
                decoder_weight = self.AttentionWeight_Final_Layer(decoder_weight_add).squeeze()
                decoder_weight = torch.min(decoder_weight, article_mask).softmax(dim=-1)

                decoder_weight_pad = decoder_weight.unsqueeze(-1).repeat([1, 1, self.lstm_size * 2])
                decoder_attention_result = article_encoder_output * decoder_weight_pad
                decoder_attention_result = torch.sum(decoder_attention_result, dim=1).unsqueeze(1)

                #####################################

                decoder_input = torch.cat([decoder_input, topic_weighted, decoder_attention_result], dim=-1)
                # decoder_input = torch.cat([decoder_input, torch.zeros([decoder_input.size()[0], 1, 1024]).cuda()],
                #                           dim=-1)
                # print(numpy.shape(decoder_input))
                # exit()

            decoder_predict_probability = torch.cat(decoder_predict_probability, dim=1)

            decoder_predict_probability = decoder_predict_probability.view(
                [decoder_predict_probability.size()[0] * decoder_predict_probability.size()[1],
                 decoder_predict_probability.size()[2]])
            abstract = abstract.view([abstract.size()[0] * abstract.size()[1]])

            loss = self.LossFunction(input=decoder_predict_probability, target=abstract)
            return loss
            # kld = 0.5 * torch.sum(torch.pow(paragraph_mean, 2) + torch.pow(paragraph_std, 2) - torch.log(
            #     1e-8 + torch.pow(paragraph_std, 2)) - 1) / (
            #               paragraph_mean.size()[0] * paragraph_mean.size()[1] * paragraph_mean.size()[2])
            #
            # return loss + kld


class VHTM(Seq2SeqWTopic):
    def __init__(self, lstm_size=128, cuda_flag=True):
        super(VHTM, self).__init__(lstm_size=lstm_size, cuda_flag=cuda_flag)


if __name__ == '__main__':
    from HistoricalVHTM.DataLoader import load_summarization

    cuda_flag = True

    train_loader, _, _, dictionary_embedding = load_summarization(paragraph_number=5)
    dictionary_embedding = torch.FloatTensor(dictionary_embedding)
    model = Seq2SeqWTopic()
    if cuda_flag: model.cuda()
    for batch_index, [batch_article, batch_article_length, batch_abstract, batch_abstract_length,
                      batch_paragraph] in enumerate(train_loader):
        print(batch_index, numpy.shape(batch_article), numpy.shape(batch_abstract), numpy.shape(batch_paragraph))
        loss = model(dictionary_embedding, batch_article, batch_article_length, batch_abstract, batch_abstract_length,
                     batch_paragraph)
        print(loss)
        exit()
