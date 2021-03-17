import os
import torch
from Model import Seq2SeqBasic
from DataLoader_Old import loader_summarization

if __name__ == '__main__':
    cuda_flag = True

    save_path = 'Result/BasicSingle'
    if not os.path.exists(save_path): os.makedirs(save_path)
    train_dataset, val_dataset, test_dataset, dictionary_embedding = loader_summarization(batch_size=1)

    seq2seqBasic = Seq2SeqBasic(dictionary_embedding, cuda_flag=cuda_flag)
    if cuda_flag: seq2seqBasic.cuda()
    optimizer = torch.optim.Adam(params=seq2seqBasic.parameters(), lr=5E-4)

    for episode_index in range(100):
        total_loss = 0.0
        with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
            for batchIndex, [batchArticle, batchAbstract, batchAbstractLabel] in enumerate(train_dataset):
                loss = seq2seqBasic(batchArticle, batchAbstract, batchAbstractLabel)
                loss_value = loss.cpu().detach().numpy()
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
