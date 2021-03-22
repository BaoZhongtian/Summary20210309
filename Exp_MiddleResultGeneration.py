import tqdm
import torch
import numpy
from Model import Seq2SeqBasic
from DataLoader import load_summarization

if __name__ == '__main__':
    cuda_flag = True

    train_dataset, val_dataset, test_dataset, dictionary_embedding = load_summarization(batch_size=16)
    dictionary_embedding = torch.FloatTensor(dictionary_embedding)

    seq2seqBasic = Seq2SeqBasic(cuda_flag=cuda_flag)

    for parameter_index in range(99, 100):
        print('Treating', parameter_index)
        load_path = 'Result/BasicSingle/Parameter-%04d.pkl' % parameter_index
        checkpoint = torch.load(load_path)
        seq2seqBasic.load_state_dict(checkpoint['ModelStateDict'])

        if cuda_flag: seq2seqBasic.cuda()

        seq2seqBasic.eval()
        predict_file = open('Result/BasicSingle/Predict-%04d.csv' % parameter_index, 'w')
        label_file = open('Result/BasicSingle/Label-%04d.csv' % parameter_index, 'w')
        for batchIndex, [batchArticle, batchArticleLength, batchAbstract, batchAbstractLength] in tqdm.tqdm(
                enumerate(train_dataset)):
            if batchIndex > 100: exit()
            decoder_predict_result = seq2seqBasic(
                dictionary_embedding, batchArticle, batchArticleLength, batchAbstract, batchAbstractLength,
                decode_flag=True)
            decoder_predict_result = decoder_predict_result.cpu().detach().numpy()
            # print(numpy.shape(decoder_predict_result))
            # exit()
            for indexX in range(len(batchAbstract)):
                for indexY in range(batchAbstractLength[indexX]):
                    if indexY != 0:
                        predict_file.write(',')
                        label_file.write(',')
                    predict_file.write(str(int(decoder_predict_result[indexX][indexY])))
                    label_file.write(str(int(batchAbstract[indexX][indexY])))
                predict_file.write('\n')
                label_file.write('\n')

            # exit()
