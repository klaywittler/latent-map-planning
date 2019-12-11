import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils # We should use this eventually.
from PIL import Image
import glob
import numbers

class CarlaDataset(Dataset):
    def __init__(self, data_dir, load_as_grayscale=False, transform=None):
        # xcxc I'm assuming that the images live in _out.
        self.data_dir = data_dir
        self.transform = transform
        self.load_as_grayscale = load_as_grayscale
        self.df = self._get_dataframe()
        
    def __len__(self):
        num_rows, _ = self.df.shape
        return num_rows
    
    def __getitem__(self, idx):
        '''
        Generate one sample of data.
        '''
        # We're gonna do some hardcore hard-coding here.
        # First, extract our control inputs
        row = self.df.iloc[idx]
        imgs_t0, imgs_t1 = self._get_image_tensor_for_row(row)
        control_inputs = np.array([row['ctr1'], row['ctr2']])
    
        return (imgs_t0, imgs_t1, control_inputs)
    
    def _get_image_tensor_for_row(self, row):
        '''
        Inputs:
            row_id: String that represents the input_num
        Outputs:
            A (2 x H x W x 4) 4D matrix of the two images.
        '''
        # The row_id should be the input_num. Should also be a string.
        img_t0_filenames = [row['img1_t0'], row['img2_t0']]
        img_t1_filenames = [row['img1_t1'], row['img2_t1']]
        img_t0_filenames.sort()
        img_t1_filenames.sort()
        t0_images = []
        for ele in img_t0_filenames:
            full_name = os.path.join(self.data_dir, '_out', ele)
            img_as_np_arr = self._load_image_and_maybe_apply_transform(full_name)
            t0_images.append(img_as_np_arr)
        t1_images = []
        for ele in img_t1_filenames:
            full_name = os.path.join(self.data_dir, '_out', ele)
            img_as_np_arr = self._load_image_and_maybe_apply_transform(full_name)
            t1_images.append(img_as_np_arr)
        
        t0_images = np.array(t0_images)
        t1_images = np.array(t1_images)
        return t0_images, t1_images

    def _load_image_and_maybe_apply_transform(self, image_loc):
        '''
        Inputs:
            image_loc: The location of the image we want to load
        Outputs:
            Either the grayscale image, a RGB image with the axes flopped, or 
            the RGB image with some series of transformations applied. 
            All are converted to numpy arrays before yeeting them out.
        '''
        # I've been writing too much haskell
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
        filename_df = self._get_image_path_df()
        df = control_input_df.merge(right=filename_df,
                                    left_on=['trajectory','input_num'],
                                    right_on=['trajectory', 'index'])
        df = df.drop_duplicates() # I have no idea why we have dupes
        
        stationary_mask = df['img1_t0'] == df['img1_t1']
        ctr1_col = df['ctr1'].copy()
        ctr2_col = df['ctr2'].copy()
        ctr1_col[stationary_mask] = 0
        ctr2_col[stationary_mask] = 0
        df['ctr1'] = ctr1_col
        df['ctr2'] = ctr2_col
        return df

    def _get_control_input_df(self):
        # xcxc I'm also assuming that our columns in control_input stay static like so.
        control_input_df = pd.read_csv(os.path.join(self.data_dir, 'control_input.txt'),
                               names=['trajectory', 'input_num', 'ctr1', 'ctr2'])
        control_input_df['input_num'] = control_input_df['input_num'].astype('int')
        control_input_df['trajectory'] = control_input_df['trajectory'].astype('str')
        return control_input_df
    
    def _get_image_path_df(self):
        filename_groupings = self._get_filename_groupings()
        filename_df = self._get_initial_filename_dataframe(filename_groupings)
        timestep_df = self._get_filename_dataframe_with_steps(filename_df)
        return timestep_df
    
    def _get_files_in_out(self):
        full_data = glob.glob(os.path.join(self.data_dir, '_out', '**.png'))
        abbrev_data = [x.split('/')[-1] for x in full_data]
        return abbrev_data
    
    def _get_filename_groupings(self):
        '''
        Reads in all the filenames, then groups them by
        (trajectory, timestep): [images]
        '''
        # A little cryptic, but it just gets the list of all filenames
        all_files_in_out = self._get_files_in_out()
        
        # We can then make a map with our data...
        filename_groupings = {}
        for fn in all_files_in_out:
            # Apologies for the hardcoding
            fn_number = str(int(fn.split('_')[0]))
            trajectory_number = str(int(fn.split('_')[2].split('.')[0]))
            if (fn_number, trajectory_number) not in filename_groupings:
                filename_groupings[(fn_number, trajectory_number)] = []
            filename_groupings[(fn_number, trajectory_number)].append(fn)
        return filename_groupings
    
    def _get_initial_filename_dataframe(self, filename_groupings):
        '''
        Given the filename groupings from the above, create a dataframe
        of the schema [trajectory, index, image1, image2]
        '''
        filename_df = pd.DataFrame(columns=['trajectory', 'index', 'img1', 'img2'])
        for k,v in filename_groupings.items():
            (index, traj) = k
            img1, img2 = None, None
            v.sort()
            if len(v) == 2:
                img1, img2 = v[0], v[1]
            elif len(v) == 1:
                img1 = v[0]
            filename_df = filename_df.append({
                'trajectory': traj,
                'index': index,
                'img1': img1,
                'img2': img2
            }, ignore_index=True)
        filename_df['trajectory'] = filename_df['trajectory'].astype('str')
        filename_df['index'] = filename_df['index'].astype('int')
        filename_df = filename_df.dropna(subset=['img1','img2']) # Drop if any of our images is None.
        return filename_df
    
    def _get_filename_dataframe_with_steps(self, filename_df):
        '''
        Given the filename df from the above, loop through it and get 
        the (t,t) and (t, t+1) pairings.
        '''
        schema = [
            'trajectory',
            'index',
            'img1_t0',
            'img2_t0',
            'img1_t1',
            'img2_t1']
        timestep_df = pd.DataFrame(columns=schema)
        num_rows, _ = filename_df.shape
        for i in range(num_rows): # blah blah yeah i know i'm not vectorizing. 
            ith_row = filename_df.iloc[i]
            row_dict = {
                'trajectory': ith_row['trajectory'],
                'index': ith_row['index'],
                'img1_t0': ith_row['img1'],
                'img2_t0': ith_row['img2']
            }
            # First, construct the stationary row
            stationary_row = row_dict.copy()
            stationary_row['img1_t1'] = ith_row['img1']
            stationary_row['img2_t1'] = ith_row['img2']
            timestep_df = timestep_df.append(stationary_row, ignore_index=True)
            # Then, construct the t+1th row IF it exists
            next_timestep_row = self._construct_next_timestep_row(
                row_dict, ith_row, filename_df)
            timestep_df = timestep_df.append(next_timestep_row, ignore_index=True)
        timestep_df['trajectory'] = timestep_df['trajectory'].astype('str')
        timestep_df['index'] = timestep_df['index'].astype('int')
        return timestep_df

    def _construct_next_timestep_row(self, row_dict, ith_row, filename_df):
        delta = 4
        next_index = ith_row['index'] + delta
        mask = (filename_df['index'] == next_index) & (filename_df['trajectory'] == ith_row['trajectory'])
        res = filename_df[mask]
        num_results = len(res)
        if num_results > 1:
            print("THERE'S MORE THAN ONE RESULT WHEN MAKING THE DATAFRAME")
        if num_results == 1:
            next_step_row = row_dict.copy()
            next_step_row['img1_t1'] = res['img1'].values[0]
            next_step_row['img2_t1'] = res['img2'].values[0]
            return next_step_row
        return None
