import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model.unwd import AGSDNetV2
from model.unwd_train import AGSDLoader, DataLoader
from lib.bilinear import getBilinear
from lib.submit import submit

if __name__ == '__main__':

    cudnn.benchmark = True
    torch.cuda.set_device(6)

    bili = getBilinear(order=0)
    # near = getBilinear(order=0)

    # for layer in range(3):
    layer = 1
    print('Generate layer %s'%layer)
    eval_dataset = AGSDLoader(mode='test', layer=layer)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    model = AGSDNetV2().cuda()
    state_dict = model.state_dict()
    for n, p in torch.load('./checkpoint/agsdDem/epoch_40.pth', map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
    result = []
    model.eval()
    with torch.no_grad():
        for data in eval_dataloader:
            inputs, _, _, sb, sn, dem, dem2 = data
            inputs, sb, sn, dem, dem2 = inputs.cuda().float(), sb.cuda().float(), sn.cuda().float(), dem.cuda().float(), dem2.cuda().float()
            with torch.no_grad():
                preds, _ = model(inputs, dem, sb, sn, dem2)
            result.append(preds.data.cpu().numpy())
    result = np.concatenate(result, axis=0).transpose((2,3,1,0))
    bili[:,:,layer:layer+1,:] = result
    # bili[:,:,2:3,:] = near[:,:,2:3,:]

    submit(bili, 'agsdDem.mat')


