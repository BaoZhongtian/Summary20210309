import torch
import numpy


class VaeOneHot(torch.nn.Module):
    def __init__(self, topic_number, accumulate_flag=False, cuda_flag=True):
        super(VaeOneHot, self).__init__()
        self.MaxWordIds = 30000
        self.AccumulateFlag = accumulate_flag
        self.CudaFlag = cuda_flag

        self.Encoder1st = torch.nn.Linear(in_features=self.MaxWordIds, out_features=5000)
        self.Encoder2nd = torch.nn.Linear(in_features=5000, out_features=500)
        self.Encoder3rd = torch.nn.Linear(in_features=500, out_features=topic_number)
        self.Decoder1st = torch.nn.Linear(in_features=topic_number, out_features=500)
        self.Decoder2nd = torch.nn.Linear(in_features=500, out_features=5000)
        self.Decoder3rd = torch.nn.Linear(in_features=5000, out_features=self.MaxWordIds)

        self.loss = torch.nn.SmoothL1Loss()

    def __generate_one_hot__(self, batch_data):
        one_hot_data = []
        for indexX in range(len(batch_data)):
            one_hot_sample = numpy.zeros(self.MaxWordIds)
            for indexY in range(len(batch_data[indexX])):
                if self.AccumulateFlag:
                    one_hot_sample[batch_data[indexX][indexY]] += 1
                else:
                    one_hot_sample[batch_data[indexX][indexY]] = 1
            one_hot_data.append(one_hot_sample)
        return torch.FloatTensor(one_hot_data)

    def forward(self, batch_data):
        one_hot_data = self.__generate_one_hot__(batch_data)
        if self.CudaFlag: one_hot_data = one_hot_data.cuda()

        encoder1st = self.Encoder1st(one_hot_data).relu()
        encoder2nd = self.Encoder2nd(encoder1st).relu()
        encoder3rd = self.Encoder3rd(encoder2nd).relu()
        decoder1st = self.Decoder1st(encoder3rd).relu()
        decoder2nd = self.Decoder2nd(decoder1st).relu()
        reconstrction = self.Decoder3rd(decoder2nd)

        return encoder3rd, self.loss(input=reconstrction, target=one_hot_data)


class VaeSeq2Seq(torch.nn.Module):
    def __init__(self, topic_number, cuda_flag=True):
        super(VaeSeq2Seq, self).__init__()
    
    def forward(self, batch_data):
        pass


if __name__ == '__main__':
    from DataLoader import loader_cnn_dm

    cuda_flag = True

    model = VaeOneHot(topic_number=100, accumulate_flag=True, cuda_flag=cuda_flag)
    if cuda_flag: model.cuda()
    train_loader, val_loader, test_loader = loader_cnn_dm(sentence_vae_flag=True)
    for batch_index, batch_data in enumerate(val_loader):
        topic, loss = model(batch_data)
        print(loss)
        exit()
