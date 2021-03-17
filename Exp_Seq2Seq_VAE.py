import os
import torch
import numpy
from Model import Seq2SeqTopicBatch
from DataLoader_Old import loader_summarization

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cuda_flag = True

    save_path = 'Result/BasicSingle-VAE-1-Part-Again'
    if not os.path.exists(save_path): os.makedirs(save_path)
    train_dataset, val_dataset, test_dataset, dictionary_embedding = loader_summarization(
        batch_size=12, topic_flag=True, topic_name='Result-VAE-1')

    seq2seqBasic = Seq2SeqTopicBatch(dictionary_embedding, cuda_flag=cuda_flag)
    if cuda_flag: seq2seqBasic = seq2seqBasic.cuda()
    optimizer = torch.optim.Adam(params=seq2seqBasic.parameters(), lr=5E-4)

    # for batchIndex, [batchArticle, batchAbstract, batchAbstractLabel, batchTopic] in enumerate(test_loader):
    #     print(batchIndex, numpy.shape(batchArticle), numpy.shape(batchAbstract), numpy.shape(batchAbstractLabel),
    #           numpy.shape(batchTopic))
    #     loss = seq2seqBasic(batchArticle, batchAbstract, batchAbstractLabel, topic=batchTopic)
    #     print(loss)
    #     exit()

    for episode_index in range(100):
        total_loss = 0.0
        with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
            for batchIndex, [batchArticle, batchAbstract, batchAbstractLabel, batchTopic] in enumerate(train_dataset):
                loss = seq2seqBasic(batchArticle, batchAbstract, batchAbstractLabel, topic=batchTopic)
                # exit()
                loss_value = loss.cpu().detach().numpy()
                if loss_value is numpy.nan: continue
                total_loss += loss_value
                print('\rBatch %d Loss = %f' % (batchIndex, loss_value), end='')
                file.write(str(loss_value) + '\n')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        print('\nEpisode %d Total Loss = %f' % (episode_index, total_loss))
        torch.save(obj={'ModelStateDict': seq2seqBasic.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
                   f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))
        torch.save(obj=seq2seqBasic, f=os.path.join(save_path, 'Network-%04d.pkl' % episode_index))
