import argparse
import copy
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from model.srcnn import SRCNN3D
import datetime
from dateutil.relativedelta import relativedelta
from lib.bilinear import otherBi
from lib.config import *
from lib.getStation import getLabel


class SRCNNLoader3D(Dataset):
    def __init__(self, mode='train'):
        super(SRCNNLoader3D, self).__init__()
        startTime = datetime.datetime(2019,4,1,0) if mode == 'train' else datetime.datetime(2019,6,30,0)
        endTime = datetime.datetime(2019,6,30,0) if mode == 'train' else datetime.datetime(2019,7,1,0)
        days = (endTime-startTime).days
        self.fileList = []
        for hour in range(days*24):
            presentTime = startTime + relativedelta(hours=hour)
            presentStr = presentTime.strftime('%Y%m%d%H')
            filePath = os.path.join('./data/nc_of_201904-201906', presentStr[:6], presentStr+'00ft.nc')
            if os.path.exists(filePath):
                self.fileList.append(filePath)
        label = getLabel()
        self.label = label[label['time'].dt.minute==0]
        self.key = ['F22', 'F3', 'F17', 'F20', 'F25', 'F8', 'F10', 'F15', 'F21']

    def __getitem__(self, item):
        path = self.fileList[item]
        with nc.Dataset(path) as fp:
            u = np.expand_dims(otherBi(fp['usig'][:][:3].data), axis=0)
            v = np.expand_dims(otherBi(fp['vsig'][:][:3].data), axis=0)
            t = np.expand_dims(otherBi(fp['tsig'][:][:3].data)/50.0, axis=0)
            h = np.expand_dims(otherBi(fp['rhsig'][:][:3].data)/100.0, axis=0)
        s = np.sqrt(u**2+v**2)
        data = np.concatenate([s, t, h], axis=0)

        currentTime = datetime.datetime.strptime(path.split('/')[-1][:10], '%Y%m%d%H')
        label = self.label[self.label['time']==currentTime]
        truth = s[:1]
        mask = np.ones((1, 3, 150, 140))
        for key in self.key:
            if len(label[label['id'] == key]['s'].values) > 0:
                truth[:1, trainId[key][0], trainId[key][1], trainId[key][2]] = label[label['id'] == key]['s'].values[0]
                mask[:1, trainId[key][0], trainId[key][1], trainId[key][2]] = mask[:1, trainId[key][0], trainId[key][1], trainId[key][2]]*7000 # why? 150*140/numStation
        return data, truth, mask

    def __len__(self):
        return len(self.fileList)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_mae(preds, labels, mask):
    return torch.sum(torch.abs(preds-labels)*(mask!=1))/torch.sum(mask!=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    model = SRCNN3D().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr*0.1}
    ], lr=args.lr)

    train_dataset = SRCNNLoader3D()
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = SRCNNLoader3D(mode='valid')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 999

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels, mask = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        torch.save(model.state_dict(), os.path.join('./checkpoint/srcnn3d/', 'epoch_{}.pth'.format(epoch)))


        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels, mask = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                preds = model(inputs)#.clamp(0.0, 1.0)
            epoch_psnr.update(calc_mae(preds, labels, mask), len(inputs))
        print('eval mae: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg < best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, mae: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join('./checkpoint/srcnn3d/', 'best.pth'))
