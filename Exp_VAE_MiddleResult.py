import os
import tqdm
import torch
import numpy
from Model import VariationalAutoEncoder
from DataLoader import loader_summarization

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda_flag = True
    paragraph_number = 3
    load_path = 'Result/VAE-%d' % paragraph_number

    train_loader, val_loader, test_loader, dictionary_embedding = loader_summarization(
        word_bag_flag=True, batch_size=64, paragraph_number=paragraph_number)

    VAE = VariationalAutoEncoder()

    checkpoint = torch.load(os.path.join(load_path, 'Parameter-0000.pkl'))
    VAE.load_state_dict(checkpoint['ModelStateDict'])

    if cuda_flag: VAE.cuda()
    VAE.eval()
    total_hidden = []
    for batchIndex, batchArticle in tqdm.tqdm(enumerate(val_loader)):
        hidden, loss = VAE(batchArticle)
        hidden = hidden.cpu().detach().numpy()
        total_hidden.extend(hidden)
        # print(numpy.shape(total_hidden))
    numpy.save('Result-VAE-%d-Val.npy' % paragraph_number, total_hidden)

    total_hidden = []
    for batchIndex, batchArticle in tqdm.tqdm(enumerate(test_loader)):
        hidden, loss = VAE(batchArticle)
        hidden = hidden.cpu().detach().numpy()
        total_hidden.extend(hidden)
        # print(numpy.shape(total_hidden))
    numpy.save('Result-VAE-%d-Test.npy' % paragraph_number, total_hidden)
