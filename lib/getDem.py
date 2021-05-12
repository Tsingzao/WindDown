import netCDF4 as nc
import numpy as np
import scipy.ndimage.interpolation


def getDemData():
    nc1 = nc.Dataset('./data/dem/ASTGTMV003_N22E113_dem.nc')
    nc2 = nc.Dataset('./data/dem/ASTGTMV003_N22E114_dem.nc')
    dem1 = nc1['ASTER_GDEM_DEM'][:].data
    dem2 = nc2['ASTER_GDEM_DEM'][:].data
    lat1 = nc1['lat'][:].data
    lat2 = nc2['lat'][:].data
    lon1 = nc1['lon'][:].data
    lon2 = nc2['lon'][:].data
    nc1.close()
    nc2.close()

    lat = lat1[1447:1873]
    lon = np.delete(np.concatenate([lon1, lon2], axis=0), 3601, axis=0)[3494:3990]
    dem1 = dem1[1447:1873,:]
    dem2 = dem2[1447:1873,:]
    dem = np.concatenate([dem1, dem2], axis=1)
    dem = np.delete(dem, 3601, axis=1)
    dem = dem[:,3494:3990]

    dem = scipy.ndimage.interpolation.zoom(dem, zoom=(140.0/426, 150.0/496))
    dem = dem.transpose((1,0))
    dem = (dem - dem.mean())/dem.std()
    np.save('./data/dem/dem.npy', dem)


def showData():
    import os
    import datetime
    import matplotlib.pyplot as plt
    from dateutil.relativedelta import relativedelta
    from lib.bilinear import otherBi
    startTime = datetime.datetime(2019,4,1,0)
    endTime = datetime.datetime(2019,7,1,0)
    days = (endTime-startTime).days
    for hour in range(days*24):
        presentTime = startTime + relativedelta(hours=hour)
        presentStr = presentTime.strftime('%Y%m%d%H')
        filePath = os.path.join('./data/nc_of_201904-201906', presentStr[:6], presentStr+'00ft.nc')
        if os.path.exists(filePath):
            with nc.Dataset(filePath) as fp:
                u = otherBi(fp['usig'][:].data)
                v = otherBi(fp['vsig'][:].data)
        s = np.sqrt(u**2+v**2)
        plt.figure()
        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.imshow(s[i])
        plt.show()
        plt.close()


if __name__ == '__main__':
    showData()