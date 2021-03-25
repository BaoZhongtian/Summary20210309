import os
import numpy


def Rouge_1(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    grams_reference = list(reference)
    grams_model = list(model)
    temp = 0
    ngram_all = len(grams_reference)
    for x in grams_reference:
        if x in grams_model: temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1


def Rouge_2(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    grams_reference = list(model)
    grams_model = list(reference)
    gram_2_model = []
    gram_2_reference = []
    temp = 0
    ngram_all = len(grams_reference) - 1
    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
    for x in gram_2_model:
        if x in gram_2_reference: temp = temp + 1
    rouge_2 = temp / ngram_all
    return rouge_2


if __name__ == '__main__':
    for file_index in range(8):
        with open('Result/Attention/Predict-%04d.csv' % file_index, 'r') as file:
            predict_data = file.readlines()
        with open('Result/Attention/Label-%04d.csv' % file_index, 'r') as file:
            label_data = file.readlines()

        total_weight_r1, total_weight_r2 = 0.0, 0.0
        for index in range(len(predict_data)):
            total_weight_r1 += Rouge_1(predict_data[index].split(','), label_data[index].split(','))
            total_weight_r2 += Rouge_2(predict_data[index].split(','), label_data[index].split(','))
        total_weight_r1 /= len(predict_data)
        total_weight_r2 /= len(predict_data)
        print(total_weight_r1, total_weight_r2)
