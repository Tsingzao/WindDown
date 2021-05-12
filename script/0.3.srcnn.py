import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model.srcnn import SRCNN
from model.srcnn_train import SRCNNLoader, DataLoader
from lib.bilinear import getBilinear
from lib.submit import submit

if __name__ == '__main__':

    cudnn.benchmark = True
    device = torch.device('cuda: 7' if torch.cuda.is_available() else 'cpu')

    eval_dataset = SRCNNLoader(mode='test')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    bili = getBilinear(order=0)
    key_value = {1:5}#, 2:5}

    for layer, epoch in key_value.items():
        print('Generate layer %s'%layer)
        model = SRCNN().to(device)
        state_dict = model.state_dict()
        for n, p in torch.load('./checkpoint/srcnn%s/epoch_%s.pth'%(layer, epoch), map_location=lambda storage, loc: storage).items():
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
        result = np.concatenate(result, axis=0).transpose((2,3,1,0))
        bili[:,:,layer:layer+1,:] = result

    submit(bili, 'srcnn_1_near.mat')


