import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils # We should use this eventually.
from PIL import Image
import numbers

class CarlaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # xcxc I'm assuming that the images live in _out.
        self.data_dir = data_dir
        self.transform = transform
        self.df = self._get_dataframe()
        self.df_as_mat = self.df.values
    
    def __len__(self):
        num_rows, _ = self.df_as_mat.shape
        return num_rows
    
    def __getitem__(self, idx):
        '''
        Generate one sample of data.
        '''
        # We're gonna do some hardcore hard-coding here.
        # First, extract our control inputs
        row = self.df_as_mat[idx, :]
        # Filter out all our numbers- assume all numbers are control inputs.
        control_inputs = np.array(
            [x for x in row if isinstance(x, numbers.Number)])
        # Then we're going to load up our images.
        # A few notes here
        # 1. There's absolutely no guarantee that there are two images.
        images = []
        for ele in row:
            if str(ele).split('.')[-1] == 'png':
                full_name = os.path.join(self.data_dir, '_out', ele)
                np_arr = np.asarray(Image.open(full_name))
                # Apply transform on each image independently.
                if self.transform:
                    np_arr = self.transform(np_arr)
                images.append(np_arr)
        images = np.array(images)
        return images, control_inputs 
        
    def _get_dataframe(self):
        control_input_df = self._get_control_input_df()
        filename_df = self._get_image_path_df()
        df = control_input_df.merge(right=filename_df,
                                    left_on='input_num',
                                    right_on='index')
        return df
    
    def _get_control_input_df(self):
        # xcxc I'm also assuming that our columns in control_input stay static like so.
        control_input_df = pd.read_csv(os.path.join(self.data_dir, 'control_input.txt'),
                               names=['input_num', 'ctr1', 'ctr2', 'ctr3'])
        control_input_df['input_num'] = control_input_df['input_num'].astype('str')
        return control_input_df
    
    def _get_image_path_df(self):
        # A little cryptic, but it just gets the list of all filenames
        all_files_in_out = [x[2] for x in os.walk(os.path.join(self.data_dir, '_out'))][0]
        # Then filter out by getting only the png files. We can remove this step if need be.
        all_files_in_out = [img_name for img_name in all_files_in_out if img_name.split('.')[1] == 'png']

        # We can then make a map with our data...
        filename_groupings = {}
        for fn in all_files_in_out:
            fn_number = str(int(fn.split('_')[0]))
            if fn_number not in filename_groupings:
                filename_groupings[fn_number] = []
            filename_groupings[fn_number].append(fn)

        # Then make a dataframe from this dictionary
        filename_df = pd.DataFrame.from_dict(
            filename_groupings, orient='index').reset_index()
        filename_df = filename_df.dropna(subset=[0,1]) # Drop if any of our images is None.
        filename_df = filename_df[filename_df['index'].astype('int') < 494] # Drop all the ones that are after 494
        return filename_df




