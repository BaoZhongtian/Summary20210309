import numpy
import os
import pytorch_pretrained_bert

if __name__ == '__main__':

    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-large-uncased')

    with open('Result/Attention/Predict-0003.csv', 'r') as file:
        predict_data = file.readlines()
    with open('Result/Attention/Label-0003.csv', 'r') as file:
        label_data = file.readlines()
    for index in range(10):
        data = [int(sample) for sample in predict_data[index].split(',')]
        print(tokenizer.convert_ids_to_tokens(data))
        data = [int(sample) for sample in label_data[index].split(',')]
        print(tokenizer.convert_ids_to_tokens(data))
        exit()
