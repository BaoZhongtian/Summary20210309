import tqdm
import torch
from HistoricalVHTM.Model import Seq2SeqWAttention
from HistoricalVHTM.DataLoader import load_summarization

if __name__ == '__main__':
    cuda_flag = True

    train_dataset, val_dataset, test_dataset, dictionary_embedding = load_summarization(batch_size=8)
    dictionary_embedding = torch.FloatTensor(dictionary_embedding)

    seq2seqBasic = Seq2SeqWAttention(cuda_flag=cuda_flag)

    for parameter_index in range(7, 8):
        print('Treating', parameter_index)
        load_path = 'Result/Attention/Parameter-%04d.pkl' % parameter_index
        checkpoint = torch.load(load_path)
        seq2seqBasic.load_state_dict(checkpoint['ModelStateDict'])

        if cuda_flag: seq2seqBasic.cuda()

        seq2seqBasic.eval()
        predict_file = open('Result/Attention/Predict-%04d-Another.csv' % parameter_index, 'w')
        label_file = open('Result/Attention/Label-%04d-Another.csv' % parameter_index, 'w')
        for batchIndex, [batchArticle, batchArticleLength, batchAbstract, batchAbstractLength] in tqdm.tqdm(
                enumerate(test_dataset)):
            # if batchIndex > 100: exit()
            decoder_predict_result = seq2seqBasic(
                dictionary_embedding, batchArticle, batchArticleLength, batchAbstract, batchAbstractLength,
                decode_flag=True)
            # exit()
            decoder_predict_result = decoder_predict_result.cpu().detach().numpy()
            # print(numpy.shape(decoder_predict_result))
            # exit()
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
