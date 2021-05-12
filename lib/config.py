import os
import numpy as np
import netCDF4 as nc
import scipy.ndimage.interpolation


root = './data/'
ncRoot = os.path.join(root, 'nc_of_201904-201906')
tsvPath = os.path.join(root, 'trainset', 'train.tsv')

testKey = {'F1':1, 'F2':2, 'F4':1, 'F5':1, 'F6':1, 'F7':0, 'F9':1, 'F11':1, 'F16':0, 'F18':2, 'F24':1, 'F26':1}
trainKey = {'F3':1, 'F8':2, 'F10':2, 'F15':2, 'F17':1, 'F20':1, 'F21':2, 'F22':0, 'F25':1}

trainPos = {'F3':[114.0941161,22.54526889], 'F8':[114.0836667,22.55606944], 'F10':[114.0095556,22.56041667],
            'F15':[114.0291667,22.572625], 'F17':[114.0531111,22.57736944], 'F20':[114.0613889,22.52222222],
            'F21':[114.0141667,22.54888889], 'F22':[114.0305556,22.51573611], 'F25':[114.057124,22.553911]}


'-----------------get bilinear pos---------------------'
data = nc.Dataset(os.path.join(ncRoot, '201906', '201906010000ft.nc'))
lat = data['Lat'][:].data
lon = data['Lon'][:].data
lat = scipy.ndimage.interpolation.zoom(lat, zoom=(10,10))
lon = scipy.ndimage.interpolation.zoom(lon, zoom=(10,10))
data.close()

trainId = {}
for key in trainPos.keys():
    tlat = np.argmin(np.abs(lat[0]-trainPos[key][1]))
    tlon = np.argmin(np.abs(lon[:,0]-trainPos[key][0]))
    trainId[key] = [trainKey[key], tlat, tlon]
