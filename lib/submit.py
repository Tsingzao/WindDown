import scipy.io as sio
from lib.config import *


def submit(result, fileName):
    submit = {}
    for key in testKey.keys():
        submit[key] = result[:,:,testKey[key]:testKey[key]+1,:]
    submit['Lat'] = lat
    submit['Lon'] = lon
    sio.savemat(os.path.join('./submit', fileName), submit)


if __name__ == '__main__':
    pass