import torch
import numpy
import pytorch_pretrained_bert
import Model

if __name__ == '__main__':
    st = '[CLS] [SEP]'
    test_input_ids = []
    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

    text = tokenizer.tokenize(st)
    print(text)
    text_id = tokenizer.convert_tokens_to_ids(text)
    print(text_id)
    test_input_ids.append(text_id)

    model = Model.BertModelRawForEmbedding.from_pretrained('bert-large-uncased')

    Input_ids = torch.tensor(test_input_ids).view(1, -1)
    # // pooled_output即为embedding = [1，分词长度, 1024]
    # print(Input_ids)
    pooled_output = model(Input_ids)
    print(numpy.shape(pooled_output))
    print(pooled_output)
    # // 然后将pooled_output降维成[分词长度, 1024], 再按行累加得到一个embedding = [1, 1024]
    # // 传入多行文本, 即得到embedding = [batch_size, 1024]
