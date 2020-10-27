import numpy as np

def _embed_matrix(size, B, start_point):
    A = np.zeros(size, dtype=np.float32)
    C = A[start_point[0]:start_point[0]+B.shape[0], start_point[1]:start_point[1]+B.shape[1], start_point[2]:start_point[2]+B.shape[2]]
    sc = C.shape
    sb = B.shape
    if sc < sb:
        # print('C:', C.shape, 'B:', B.shape, 'start:', start_point)
        B = B[0:sc[0],0:sc[1],0:sc[2]]
    np.add(C, B, C)
    return A

# Create a 3D numpy array simulating 3D gaussian dist.
def gaussian3D(shape, sigma=1):
    r3, r2, r1 = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-r3:r3+1,-r2:r2+1,-r1:r1+1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gen_3d_heatmap(size, gt_boxes, scale=(4,4,4)):
    size = [size[i]//scale[i] for i in range(3)]
    hm = np.zeros(size, dtype=np.float32)

    for bbox in gt_boxes:
        full_width = (int(bbox['z_range']//scale[0]), int(bbox['y_range']//scale[1]), int(bbox['x_range']//scale[2]))
        # min width must be at least 1
        full_width = [w if w > 0 else 1 for w in full_width]
        expand = map(lambda x: x+1 if x%2 == 0 else x, full_width)
        full_width = list(expand)
        start_point = [int(bbox['z_bot']//scale[0]), int(bbox['y_bot']//scale[1]), int(bbox['x_bot']//scale[2])]

        if full_width[0]<11:
            full_width[0] = 11
            start_point[0] = max(0, int(bbox['z_center']//scale[0]) - 5)
        if full_width[2]<11:
            full_width[2] = 11
            start_point[2] = max(0, int(bbox['x_center']//scale[2]) - 5)
        if full_width[1]<5:
            full_width[1] = 5
            start_point[1] = max(0, int(bbox['y_center']//scale[1]) - 2)

        gauss_3d = gaussian3D(full_width, sigma=3)
        layer = _embed_matrix(size, gauss_3d, start_point)
        hm = np.maximum(hm, layer)

    return hm

def gen_3d_hw(size, gt_boxes, scale=(4,4,4)):
    size = [size[i]//scale[i] for i in range(3)]
    hw_x = np.zeros(size, dtype=np.float32)
    hw_y = np.zeros(size, dtype=np.float32)
    hw_z = np.zeros(size, dtype=np.float32)

    for bbox in gt_boxes:
        ori_shape = (int(bbox['z_range']), int(bbox['y_range']), int(bbox['x_range']))
        shape = (int(bbox['z_range']//scale[0]), int(bbox['y_range']//scale[1]), int(bbox['x_range']//scale[2]))
        start_point = [int(bbox['z_bot']//scale[0]), int(bbox['y_bot']//scale[1]), int(bbox['x_bot']//scale[2])]
        chunk = np.full(shape, ori_shape[2], dtype=np.float32)
        layer = _embed_matrix(size, chunk, start_point)
        hw_x = np.maximum(hw_x, layer)

        chunk = np.full(shape, ori_shape[1], dtype=np.float32)
        layer = _embed_matrix(size, chunk, start_point)
        hw_y = np.maximum(hw_y, layer)

        chunk = np.full(shape, ori_shape[0], dtype=np.float32)
        layer = _embed_matrix(size, chunk, start_point)
        hw_z = np.maximum(hw_z, layer)

    return hw_x, hw_y, hw_z
