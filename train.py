from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import NetWork
import torch.optim as optim
import torch.nn as nn
import torch

from torch.autograd import Variable

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

dataset = Imgdataset("/home/tenshine/Desktop/SRCNN/data")
train_data_loader = DataLoader(dataset=dataset, num_workers=10, batch_size=20, shuffle=True)

net = NetWork()
loss = nn.MSELoss()
net.cuda()
loss.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.01)

def train(epoch):
    epoch_loss = 0
    for iteration,  batch in enumerate(train_data_loader):
        input, target = Variable(batch[0]), Variable(batch[1])
        input = input.cuda()
        target = target.cuda()
        target = target[:, :, 6:26, 6:26]
        optimizer.zero_grad()
        # print ("input shape = " , input.shape)
        # print ("target shape = ", target.shape)
        model_out = net(input)

        # print ("model_out shape =" , model_out.shape)
        Loss = loss(model_out, target)
        # print('Loss data', Loss.data)
        epoch_loss += Loss.data
        Loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_data_loader), Loss.data))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = srcnn(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, 201):
    print('epoch starts')
    train(epoch)
    # test()
    if(epoch%10==0):
        checkpoint(epoch)


