import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom

def loadFileInformation(filename):
    information = {}
    ds = dicom.read_file(filename)

    information['NumberOfFrames'] = ds.NumberOfFrames if 'NumberOfFrames' in dir(ds) else 1
    information['PixelSpacing'] = ds.PixelSpacing if 'PixelSpacing' in dir(ds) else [1, 1]
    information['Rows'] = ds.Rows
    information['Columns'] = ds.Columns
    information['SliceThickness'] = ds.SliceThickness
    information['SpacingBetweenSlices'] = ds.SpacingBetweenSlices

    data = ds.pixel_array

    return [float(ds.SpacingBetweenSlices), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]


def AUC(froc_x, froc_y, normalize=False):
    froc_x = np.array(froc_x)
    froc_y = np.array(froc_y)

    area = np.trapz(froc_y[::-1], x=froc_x[::-1], dx=0.001)

    if normalize:
        return area/np.max(froc_x[::-1])
    else:
        return area


def draw_full(froc_x, froc_y, color, label, linestyle, x_limit, normalize=False):
    area = AUC(froc_x, froc_y, normalize=normalize)
    plt.plot(froc_x, froc_y, color=color, label=label +
             ', Az = %.3f' % area, linestyle=linestyle)

def build_threshold(th_step):
    thresholds = []

    tmp=0
    steps = int(0.9875 // th_step)
    for i in range(0, steps+1):
        thresholds.append(tmp)
        tmp += th_step
    # for i in range(0, 75):
    #     thresholds.append(tmp)
    #     tmp += 0.01

    return thresholds


def categorize_by_size(box):
    z0, y0, x0, z1, y1, x1 = box
    z, y, x = (z1-z0)//4, (y1-y0)//4, (x1-x0)//4
    max_axis = max(z, y, x)
    if max_axis <= 10:
        return (1,0,0)
    elif max_axis >= 15:
        return (0,0,1)
    else:
        return (0,1,0)


