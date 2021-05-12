from lib.bilinear import getBilinear
from lib.submit import submit


# submit(getBilinear(order=3), 'bilinear.mat')
submit(getBilinear(order=0), 'nearest.mat')