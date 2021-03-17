import os
import tqdm
import pytorch_pretrained_bert

if __name__ == '__main__':
    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-large-uncased')

    data = '[CLS] I eat an apple.'

    tokenized_text = tokenizer.tokenize(data)
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    print(indexed_tokens)
