import argparse
import copy
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from model.unwd import AGSDNetV2
import datetime
from dateutil.relativedelta import relativedelta
from lib.bilinear import otherBi
from lib.config import *
from lib.getStation import getLabel
import random

seed = 508
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class AGSDLoader(Dataset):
    def __init__(self, mode='train', layer=1):
        super(AGSDLoader, self).__init__()
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
        key = [['F22'], ['F3', 'F17', 'F20', 'F25'], ['F8', 'F10', 'F15', 'F21']]
        self.key = key[layer]
        self.layer = layer
        self.dem1 = np.expand_dims(scipy.ndimage.interpolation.zoom(np.load('./data/dem/dem.npy'), zoom=(0.1,0.1)), axis=0)
        self.dem2 = np.expand_dims(np.load('./data/dem/dem.npy'), axis=0)

    def __getitem__(self, item):
        path = self.fileList[item]
        with nc.Dataset(path) as fp:
            u = fp['usig'][:][self.layer:self.layer+1].data
            v = fp['vsig'][:][self.layer:self.layer+1].data
            t = fp['tsig'][:][self.layer:self.layer+1].data/50.0
            h = fp['rhsig'][:][self.layer:self.layer+1].data/100.0
            p = (fp['psig'][:][self.layer:self.layer+1].data-900)/100.0
        ub = otherBi(u)
        vb = otherBi(v)
        un = otherBi(u, order=0)
        vn = otherBi(v, order=0)
        s = np.sqrt(u**2+v**2)
        data = np.concatenate([s, t, h, p], axis=0)
        sb = np.sqrt(ub**2+vb**2)
        sn = np.sqrt(un**2+vn**2)

        currentTime = datetime.datetime.strptime(path.split('/')[-1][:10], '%Y%m%d%H')
        label = self.label[self.label['time']==currentTime]
        truth = sb # sn #
        mask = np.ones((1, 150, 140))
        for key in self.key:
            if len(label[label['id'] == key]['s'].values) > 0:
                truth[0, trainId[key][1], trainId[key][2]] = label[label['id'] == key]['s'].values[0]
                mask[0, trainId[key][1], trainId[key][2]] = mask[0, trainId[key][1], trainId[key][2]]*6000 # why? 150*140/numStation
        return data, truth, mask, sb, sn, self.dem1, self.dem2

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
    parser.add_argument('--num-epochs', type=int, default=100)
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--gpuID', type=int, default=7)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpuID)
    cudnn.benchmark = True
    # device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    model = AGSDNetV2().cuda().float()
    criterion = nn.MSELoss()
    criterion_= nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_dataset = AGSDLoader(layer=args.layer)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = AGSDLoader(mode='valid', layer=args.layer)
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
                inputs, labels, mask, sb, sn, dem1, dem2 = data

                inputs = inputs.cuda().float()
                labels = labels.cuda().float()
                mask = mask.cuda().float()
                sb = sb.cuda().float()
                sn = sn.cuda().float()
                dem1 = dem1.cuda().float()
                dem2 = dem2.cuda().float()

                preds, aux = model(inputs, dem1, sb, sn, dem2)
                loss = criterion(preds, labels)
                loss_= criterion_(aux, dem2)
                loss = loss + loss_*5

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        torch.save(model.state_dict(), os.path.join('./checkpoint/agsdDem/', 'epoch_{}.pth'.format(epoch)))
        scheduler.step()

        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels, mask, sb, sn, dem1, dem2 = data
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()
            mask = mask.cuda().float()
            sb = sb.cuda().float()
            sn = sn.cuda().float()
            dem1 = dem1.cuda().float()
            dem2 = dem2.cuda().float()
            with torch.no_grad():
                preds, aux = model(inputs, dem1, sb, sn, dem2)#.clamp(0.0, 1.0)
            epoch_psnr.update(calc_mae(preds, labels, mask), len(inputs))
        print('eval mae: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg < best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, mae: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join('./checkpoint/agsdDem/', 'best.pth'))
