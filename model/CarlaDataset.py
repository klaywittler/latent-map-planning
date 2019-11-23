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
        # xcxc We're... We're not exactly doing anything with our control inputs. For now.
        control_inputs = np.array(
            [x for x in row if isinstance(x, numbers.Number)])
        
        curr_images = self._get_image_tensor_for_row(row[0])
        # Get the next row
        next_delta = 4 # xcxc This is a hardcoded parameter from Klayton's data.
        next_input_id = int(row[0]) + next_delta
        num_rows_next = np.sum(self.df['input_num'] == str(next_input_id))
        if num_rows_next == 0:
            # No next: treat it as if we're stationary
            return (curr_images, curr_images, np.zeros(len(control_inputs)))
        else:
            next_images = self._get_image_tensor_for_row(str(next_input_id))
            return (curr_images, next_images, control_inputs)
    
    def _get_image_tensor_for_row(self, row_id):
        '''
        Inputs:
            row_id: String that represents the input_num
        Outputs:
            A (2 x H x W x 4) 4D matrix of the two images.
        '''
        # The row_id should be the input_num. Should also be a string.
        row = self.df[self.df['input_num'] == row_id]
        n_res, _ = row.shape
        if n_res > 1:
            # xcxc I'm assuming there's only one row per row_id.
            # This may or may not be a strictly held invariant.
            print("XCXC: THERE ARE MORE THAN 1 ROW FOR A ROW_ID")
        row = row.values[0]
        images = []
        for ele in row:
            if str(ele).split('.')[-1] == 'png':
                full_name = os.path.join(self.data_dir, '_out', ele)
                np_arr = np.asarray(Image.open(full_name))
                np_arr = self._rearrange_axes_image(np_arr)
                # Apply transform on each image independently.
                if self.transform:
                    np_arr = self.transform(np_arr)
                images.append(np_arr)
        images = np.array(images)
        return images
    
    def _rearrange_axes_image(self, img):
        H,W,_ = img.shape
        new_img = np.zeros((3,H,W))
        for i in range(3):
            new_img[i,:,:] = img[:,:,i]
        return new_img

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
                               names=['input_num', 'ctr1', 'ctr2'])
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
#         filename_df = filename_df[filename_df['index'].astype('int') < 494] # Drop all the ones that are after 494
        return filename_df
