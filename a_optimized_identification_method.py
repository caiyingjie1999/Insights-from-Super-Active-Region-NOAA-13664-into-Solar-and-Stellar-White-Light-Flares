import os
import sunpy.map
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import label


def get_filepath(path):
    file = os.listdir(path)
    file = sorted(map(str, file))
    image_path = []
    for i in range(len(file)):
        p = path+'/'+str(file[i])
        image_path.append(p)
    return file, image_path

def find_data(row, column, delta, row0, column0):
    lookup = {(r, c): d for r, c, d in zip(row, column, delta)}
    delta0 = [lookup.get((r, c), None) for r, c in zip(row0, column0)]
    return delta0

def filter_connected_regions(row, column):
    max_row = np.max(row) + 1
    max_col = np.max(column) + 1
    data = np.zeros((max_row, max_col), dtype=int)
    data[row, column] = 1
    labeled_array, num_features = label(data)
    sizes = [(np.sum(labeled_array == i), i) for i in range(1, num_features + 1)]
    selected_region = []
    for i in range(len(sizes)):
        if sizes[i][0] > 8:
            selected_region.append(sizes[i][1])
    number_of_selected_region = len(selected_region)
    largest_regions = sorted(sizes, reverse=True)[:number_of_selected_region]
    row1 = []
    column1 = []
    for size, labell in largest_regions:
        region_coords = np.argwhere(labeled_array == labell)
        r, c = region_coords[:, 0], region_coords[:, 1]
        row1.extend(r)
        column1.extend(c)
    return row1, column1


num = 1
coefficient = 1
limit = 9

path_file = r'G:\paper1\data\{}\remove_dff'.format(num)
hmi_file,hmi_image_path = get_filepath(path_file)

path_csv = r'G:\paper1\data\{}\flare_region.csv'.format(num)
row0, column0 = pd.read_csv(path_csv).iloc[:, :2].values.T.tolist()

path_delta = r'G:\paper1\data\{}\delta\0.csv'.format(num)
row, column, delta = pd.read_csv(path_delta).iloc[:, :3].values.T.tolist()
delta0 = find_data(row, column, delta, row0, column0)

filtered_data = [(r, c, d) for r, c, d in zip(row0, column0, delta0) if d is not None]
row0, column0, delta0 = zip(*filtered_data)
row0, column0, delta0 = [list(x) for x in (row0, column0, delta0)]


row3 = []
column3 = []
# 选取耀斑带时间段内的数据
for xx in range(40, len(hmi_file)-80):
    row1 = []
    column1 = []
    hdu_pro = fits.open(hmi_image_path[xx-1])
    adu_pro = hdu_pro[0].data 
    hdu_pro.close()
    hdu_now = fits.open(hmi_image_path[xx])
    adu_now = hdu_now[0].data
    hdu_now.close()
    hdu_back = fits.open(hmi_image_path[xx+1])
    adu_back = hdu_back[0].data 
    hdu_back.close()

    for i,j,q in zip(row0,column0,delta0): 
        f = 0 
        try:
            for m in range(i-1,i+2):
                for n in range(j-1,j+2):
                    value1 = abs(adu_now[m][n]-adu_pro[m][n])/(adu_pro[m][n])
                    value2 = abs(adu_back[m][n]-adu_now[m][n])/(adu_now[m][n])
                    if value1 >=  coefficient*q:
                        f = f+1
                    else:  
                        pass
                    if value2 >=  coefficient*q:
                        f = f+1
                    else:  
                        pass
        except IndexError:
            pass
        if f >= limit:
            row1.append(i)
            column1.append(j)
        else:
            pass
    try:
        row2, column2 = filter_connected_regions(row1, column1)
    except ValueError:
        row2 = []
        column2 = []
    row3.extend(row2)
    column3.extend(column2)

coordinates = list(set(zip(row3, column3)))
row4, column4 = zip(*coordinates)  
row4, column4 = list(row4), list(column4)

omap = sunpy.map.Map(hmi_image_path[0])
xscale = (omap.data).shape[0]
empty_array = np.zeros((xscale, xscale), dtype=int)
for i,j in zip(row0,column0):
    empty_array[i][j] = 10000
fig,ax = plt.subplots(dpi=200, subplot_kw={'projection': omap})
ax.imshow(empty_array, cmap='gray')
omap.plot(axes=ax)
ax.set_title('')
plt.scatter(column4, row4, s=2, linewidths=0.1, alpha=1, marker='s', c='skyblue')
# 绘制等高线图
y, x = np.indices(empty_array.shape)
contour = ax.contour(x, y, empty_array, colors='r', linewidths=0.1)
plt.xlabel('Solar X (arcsec)')
plt.ylabel('Solar Y (arcsec)')
plt.show()