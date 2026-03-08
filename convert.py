import os
import numpy as np
import h5py
from PIL import Image

data_root = 'data/figshare'
output_folder = 'data/figshare_images'

subfolders = [
    'brainTumorDataPublic_1-766',
    'brainTumorDataPublic_767-1532',
    'brainTumorDataPublic_1533-2298',
    'brainTumorDataPublic_2299-3064',
]

for subfolder in subfolders:
    input_folder = os.path.join(data_root, subfolder)
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    print(f'Processing {subfolder} — {len(file_list)} files')

    out_subfolder = os.path.join(output_folder, subfolder)
    os.makedirs(out_subfolder, exist_ok=True)

    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)

        with h5py.File(file_path, 'r') as f:
            cjdata = f['cjdata']
            im = cjdata['image'][()].astype(np.float64)

        min_val, max_val = im.min(), im.max()
        im_normalized = np.uint8(255 / (max_val - min_val) * (im - min_val))

        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(out_subfolder, base_name + '.jpg')
        Image.fromarray(im_normalized).save(output_path)

print('Done!')