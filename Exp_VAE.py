import os
import numpy
import torch
from Model import VariationalAutoEncoder
from DataLoader import loader_summarization

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda_flag = True
    save_path = 'Result/VAE-5'
    if not os.path.exists(save_path): os.makedirs(save_path)

    train_loader, val_loader, test_loader, dictionary_embedding = loader_summarization(
        word_bag_flag=True, batch_size=64, paragraph_number=5)
    VAE = VariationalAutoEncoder()
    if cuda_flag: VAE.cuda()
    optimizer = torch.optim.Adam(params=VAE.parameters(), lr=5E-4)

    for episode_index in range(1):
        total_loss = 0.0
        with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
            for batchIndex, batchArticle in enumerate(train_loader):
                hidden, loss = VAE(batchArticle)
                loss = loss * 100

                loss_value = loss.cpu().detach().numpy()
                total_loss += loss_value
                print('\rBatch %d Loss = %f' % (batchIndex, loss_value), end='')
                file.write(str(loss_value) + '\n')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print('\nEpisode %d Total Loss = %f' % (episode_index, total_loss))
            torch.save(
                obj={'ModelStateDict': VAE.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
                f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))
            torch.save(obj=VAE, f=os.path.join(save_path, 'Network-%04d.pkl' % episode_index))
