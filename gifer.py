import os
from PIL import Image
import re

input_folder = 'Patch3_ConnectTrue_complete'
output_gif = 'output.gif'

png_files = [f for f in os.listdir(input_folder) if re.match(r'linLin_(\d+)_.*\.png', f)]
png_files = sorted([f for f in png_files if int(re.match(r'linLin_(\d+)_.*\.png', f).group(1)) <= 20], key=lambda x: int(re.match(r'linLin_(\d+)_.*\.png', x).group(1)))

frames = []
for png_file in png_files:
    image = Image.open(os.path.join(input_folder, png_file))
    frames.append(image)

frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=1000, loop=0)
