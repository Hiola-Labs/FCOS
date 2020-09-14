
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
def visualize(img_data):
    views = ['cor']
    for v in views:
        if not os.path.exists('./visualize/'):
            os.makedirs('./visualize/')

        img_dir = os.path.join('./visualize/', v)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if v is 'cor':
            for i in range(np.shape(img_data)[1]):
                img = Image.fromarray(img_data[:,i,:], 'L')
                img = img.convert(mode='RGB')
                draw = ImageDraw.Draw(img)
                if (0):
                    for bx in boxes:
                        z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], bx['x_bot'], bx['x_top']
                        if bx['score'] < score_threshold:
                            continue

                        if int(y_bot) <= i <= int(y_top):
                            draw.rectangle(
                                [(z_bot,x_bot),(z_top,x_top)],
                                outline ="blue", width=2)

                    for bx in gt_boxes:
                        z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]

                        if int(y_bot) <= i <= int(y_top):
                            draw.rectangle(
                                [(z_bot,x_bot),(z_top,x_top)],
                                outline ="red", width=2)
                draw.rectangle((0, 2, 30, 12), fill='blue')
                draw.text((2,0), 'Pred', fill="white")
                draw.rectangle((0, 14, 30, 24), fill='red')
                draw.text((2,14), 'GT', fill="white")

                img.save(os.path.join(img_dir, (str(i)+'.png')))