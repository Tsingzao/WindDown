import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model.srcnn import SRCNN3D
from model.srcnn3d_train import SRCNNLoader3D, DataLoader
from lib.bilinear import getBilinear
from lib.submit import submit

if __name__ == '__main__':

    cudnn.benchmark = True
    device = torch.device('cuda: 7' if torch.cuda.is_available() else 'cpu')

    eval_dataset = SRCNNLoader3D(mode='test')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    bili = getBilinear()

    model = SRCNN3D().to(device)
    state_dict = model.state_dict()
    for n, p in torch.load('./checkpoint/srcnn3d/epoch_%s.pth'%(4), map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
    result = []
    model.eval()
    with torch.no_grad():
        for data in eval_dataloader:
            inputs, _, _ = data
            inputs = inputs.to(device)
            with torch.no_grad():
                preds = model(inputs)
            result.append(preds.data.cpu().numpy())
    result = np.concatenate(result, axis=0).squeeze(1).transpose((2,3,1,0))
    bili[:,:,:3,:] = result

    submit(bili, 'srcnn3d.mat')


