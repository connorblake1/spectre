from PIL import Image
import numpy as np
# MICKEY MOUSE PHASE INVERSION - JUST INVERT THE RED AND BLUE PIXELS
folder = "Patch2_ConnectTrue_good"
filename = "linLin_19_e_-2.386_"
image_path = folder + "\\" + filename + ".png"
image = Image.open(image_path)
image_array = np.array(image)
rowlen = image_array.shape[1]
for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        if j > .80*rowlen: # ie where the color bar roughly is
            break
        pixel = image_array[i, j]
        r,g,b,a = pixel
        if r != b:
            image_array[i, j] = (b,g,r,a)
inverted_image = Image.fromarray(image_array)
inverted_image.save(folder + "\\" + filename+ "_Inverted.png")