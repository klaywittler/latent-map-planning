'''
A variant on the original data loader

'''
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils # We should use this eventually.
from PIL import Image
import numbers

class VelocityPredictionCarlaDataSet(Dataset):
    def __init__(self, data_dir, goal_image, delta=10, load_as_grayscale=False, transform=None):
        # xcxc I'm assuming that the images live in _out.
        self.data_dir = data_dir
        self.transform = transform
        if goal_image == None:
            print("WARNING: MUST PROVIDE GOAL IMAGE.")
        self.goal_image = goal_image
        self.delta = delta
        self.load_as_grayscale = load_as_grayscale
        self.df = self._get_dataframe()
    
    def __len__(self):
        num_rows, _ = self.df.shape
        return num_rows
    
    def __getitem__(self, idx):
        '''
        Generate one sample of data.
        '''
        row = self.df.iloc[idx]
        ctr1 = float(row['ctr1'])
        ctr2 = float(row['ctr2'])
        control_inputs = np.array([ctr1, ctr2])
        src_img = self._load_image_and_maybe_apply_transform(row['src'])
        tgt_img = self._load_image_and_maybe_apply_transform(row['tgt'])
        return (src_img, tgt_img, control_inputs)

    def _load_image_and_maybe_apply_transform(self, filename):
        '''
        Inputs:
            image_loc: The location of the image we want to load
        Outputs:
            Either the grayscale image, a RGB image with the axes flopped, or 
            the RGB image with some series of transformations applied. 
            All are converted to numpy arrays before yeeting them out.
        '''
        # I've been writing too much haskell
        image_loc = os.path.join(self.data_dir, '_out', filename)
        pil_img = Image.open(image_loc)
        if self.load_as_grayscale:
            pil_img = pil_img.convert('L')
        
        if self.transform:
            transform_result = self.transform(pil_img)
            return np.asarray(transform_result[:3, :, :])
        else:
            if self.load_as_grayscale:
                return np.array(pil_img)
            else:
                return self._rearrange_axes_image(np.array(pil_img))

    def _rearrange_axes_image(self, img):
        H,W,_ = img.shape
        new_img = np.zeros((3,H,W))
        for i in range(3):
            new_img[i,:,:] = img[:,:,i]
        return new_img

    def _get_dataframe(self):
        control_input_df = self._get_control_input_df()
        control_input_df['input_num'] = control_input_df['input_num'].astype('int') 
        filename_df = self._get_image_path_df()
        filename_df['index'] = filename_df['index'].astype('int')
        df = control_input_df.merge(right=filename_df,
                                    left_on='input_num',
                                    right_on='index')
        stationary_mask = (df['src'] == df['tgt'])
        ctr1_col = df['ctr1'].copy()
        ctr2_col = df['ctr2'].copy()
        ctr1_col[stationary_mask] = 0
        ctr2_col[stationary_mask] = 0
        df['ctr1'] = ctr1_col
        df['ctr2'] = ctr2_col
        df = df[['ctr1', 'ctr2', 'src', 'tgt']]
        return df

    def _get_control_input_df(self):
        # xcxc I'm also assuming that our columns in control_input stay static like so.
        control_input_df = pd.read_csv(os.path.join(self.data_dir, 'control_input.txt'),
                               names=['input_num', 'ctr1', 'ctr2'])
        control_input_df['input_num'] = control_input_df['input_num'].astype('str')
        return control_input_df
    
    def _get_image_path_df(self):
        '''
        Different from the OG CarlaDS.
        This returns a dataframe of the 
        '''
        all_files_in_out = self._get_image_files_in_directory()
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
        filename_df = filename_df.rename(columns={0: "src"})[['index', 'src']] # Project to just get our source images.
        filename_df = self._get_pairwise_combinations(filename_df)
        return filename_df
    
    def _get_pairwise_combinations(self, df):
        '''
        With filename_df, we construct the ('index', 'src', 'tgt' here), constructed by 
        '''
        pairwise_df = pd.DataFrame(columns=['index', 'src', 'tgt'])
        num_rows, _ = df.shape
        tgt_index = int(self.goal_image.split('_')[0]) # Get which # image we want to go up to
        
        for i in range(num_rows):
            # Get data from our current row
            ith_row = df.iloc[i]
            index = int(ith_row['index'])
            # Get all the potential target images
            src_filename = ith_row['src']
            indices = list(np.arange(index, tgt_index, self.delta * 4)) # Hardcoding in 4 because images increment by 4
            if self.delta != 1:
                indices.append(index + 4) # And to get t+1 as well.
            tgt_rows = df[df['index'].astype('int').isin(indices)] # Get all the target rows
            # Then loop through our filenames and pair them together and append them to our df
            for tgt_filename in tgt_rows['src']:
                pairwise_df = pairwise_df.append({
                    'index': index,
                    'src': src_filename,
                    'tgt': tgt_filename
                }, ignore_index=True)
        return pairwise_df
    
    def _get_image_files_in_directory(self, end='png'):
        '''
        Retrieves all the filenames in the data directory with some end extension.
        Currently, end is png.
        '''
        # A little cryptic, but it just gets the list of all filenames
        all_files_in_out = [x[2] for x in os.walk(os.path.join(self.data_dir, '_out'))][0]
        # Then filter out by getting only the png files. We can remove this step if need be.
        all_files_in_out = [img_name for img_name in all_files_in_out if img_name.split('.')[1] == end]
        return all_files_in_out
    