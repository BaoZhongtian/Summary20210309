import os
import torch
from Model import VaeOneHot
from DataLoader import loader_cnn_dm

if __name__ == '__main__':
    cuda_flag = True

    save_path = 'Result/VAE_OneHot'
    if not os.path.exists(save_path): os.makedirs(save_path)
    train_dataset, val_dataset, test_dataset = loader_cnn_dm(
        batch_size=32, sentence_vae_flag=True)

    model = VaeOneHot(topic_number=50, accumulate_flag=True, cuda_flag=cuda_flag)
    if cuda_flag: model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1E-3)

    for episode_index in range(100):
        total_loss = 0.0
        with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
            for batchIndex, batchData in enumerate(train_dataset):
                _, loss = model(batchData)
                loss = loss * 1000
                loss_value = loss.cpu().detach().numpy()
                total_loss += loss_value
                print('\rBatch %d Loss = %f' % (batchIndex, loss_value), end='')
                file.write(str(loss_value) + '\n')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batchIndex % 100000 == 99999:
                    torch.save(obj={'ModelStateDict': model.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
                               f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))
                    torch.save(obj=model, f=os.path.join(save_path, 'Network-%04d.pkl' % episode_index))

        print('\nEpisode %d Total Loss = %f' % (episode_index, total_loss))
        torch.save(obj={'ModelStateDict': model.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
                   f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))
        torch.save(obj=model, f=os.path.join(save_path, 'Network-%04d.pkl' % episode_index))
