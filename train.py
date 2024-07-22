from dataset import ImgDenoiseDataset, TestDataset
import os
import torch
import argparse
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNet, Net, DnCNN
from sklearn.model_selection import train_test_split
from utils import PSNR
import numpy as np
from torch.nn.modules.loss import _Loss
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

trainset = ImgDenoiseDataset("data/train", sigma=15)
trainset, valset = train_test_split(trainset, test_size=0.2, shuffle=False)
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
val_loder = DataLoader(valset, batch_size=1, shuffle=False)
testset = TestDataset("data/test", sigma=15)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
device = "cuda"
lr = 1e-3
epochs = 100


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def train():
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    model = DnCNN().to(device)
    start_epoch = -1
    criterion = sum_squared_error()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = 99999.0
    x_axis = []
    loss_axis = []
    dataset_len = len(trainset)

    for epoch in range(start_epoch + 1, epochs):
        step = 0
        runnin_loss = 0.0
        pbar = tqdm(train_loader)


        model.train()
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            noise = inputs - targets
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, noise)
            loss.backward()
            optimizer.step()
            runnin_loss += loss.item()
            step += 1
            pbar.set_description("Epoch:[%d/ %d] Average running loss %.4f" % (epoch,
                                                                               epochs,
                                                                               (runnin_loss / step)))

        # validation
        psnr_list = []
        pbar_val = tqdm(val_loder)
        model.eval()
        iter = 0
        valid_loss = 0
        for inputs, targets in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            noise = inputs - targets
            with torch.no_grad():
                infer = model(inputs)
                loss = criterion(infer, noise)
            pnsr_value = compare_psnr(targets.squeeze(0).cpu().numpy(),
                                      (inputs - infer).squeeze(0).cpu().numpy()).item()
            psnr_list.append(pnsr_value)
            valid_loss += loss.item()
            iter += 1
            pbar_val.set_description("Epoch:[%d/ %d] Average running loss %.4f" % (epoch,
                                                                                   epochs,
                                                                                   (valid_loss / iter)))

        psnr_mean = np.mean(psnr_list)
        print('The PSNR Value is:', psnr_mean)

        if best_loss>=valid_loss / iter:
            best_loss =valid_loss / iter
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if not os.path.isdir("./logs/checkpoint"):
                os.mkdir("./logs/checkpoint")
            torch.save(checkpoint, './logs/checkpoint/best_model.pth')
            print("model saved!")
        x_axis.append(epoch)
        loss_axis.append(float(valid_loss / iter))
        plt.figure()
        plt.title("loss")
        plt.plot(x_axis, loss_axis)
        plt.savefig("./logs/loss.png")
    return model


def test(model):
    topil = transforms.ToPILImage()
    pbar_test = tqdm(test_loader)
    i = 0
    for inputs, targets in pbar_test:
        i += 1
        inputs, targets = inputs.to(device), targets.to(device)
        model.eval()
        with torch.no_grad():
            output = (model(inputs)).squeeze(0)
            noisy_input = (inputs.squeeze(0))
            pbar_test.set_description("testin")
            pic_input = topil(noisy_input)
            pic_result = topil(noisy_input - output)
            pic_result.save(f"result_{i}.jpg")
            pic_input.save(f"input_{i}.jpg")


if __name__ == '__main__':
    trained_model = train()
    test(trained_model)
