import datetime
from dateutil.relativedelta import relativedelta
from lib.config import *

def otherBi(data, order=3):
    return scipy.ndimage.interpolation.zoom(data, zoom=(1,10,10), order=order)

def getBilinear(ratio=0.5, order=3):
    startTime = datetime.datetime(2019,6,30,0)
    endTime = datetime.datetime(2019,7,1,0)
    days = (endTime-startTime).days

    result = []
    for hour in range(24*days):
        presentTime = startTime + relativedelta(hours=hour)
        presentStr = presentTime.strftime('%Y%m%d%H')
        ncPath = os.path.join(ncRoot, presentStr[:6], presentStr+'00ft.nc')
        data = nc.Dataset(ncPath)
        u = data['usig'][:].data
        v = data['vsig'][:].data
        u = scipy.ndimage.interpolation.zoom(u, zoom=(1,10,10), order=order)
        v = scipy.ndimage.interpolation.zoom(v, zoom=(1,10,10), order=order)
        s = np.sqrt(u**2+v**2)
        s = np.expand_dims(s*ratio, axis=0)
        result.append(s)
        print(presentTime)
    result = np.concatenate(result, axis=0)
    result = result.transpose((2,3,1,0))

    return result