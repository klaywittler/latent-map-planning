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
    def __init__(self, data_dir, goal_images={}, delta=100, load_as_grayscale=False, transform=None):
        # xcxc I'm assuming that the images live in _out.
        self.data_dir = data_dir
        self.transform = transform
        self.goal_images = goal_images
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
        pairwise_df = self._get_pairwise_df(filename_df)
        pairwise_df['index'] = pairwise_df['index'].astype('int')
        df = control_input_df.merge(right=pairwise_df,
                                    left_on=['input_num', 'trajectory'],
                                    right_on=['index', 'trajectory'])
        stationary_mask = (df['src'] == df['tgt'])
        ctr1_col = df['ctr1'].copy()
        ctr2_col = df['ctr2'].copy()
        ctr1_col[stationary_mask] = 0
        ctr2_col[stationary_mask] = 0
        df['ctr1'] = ctr1_col
        df['ctr2'] = ctr2_col
        df = df[['trajectory', 'index', 'ctr1', 'ctr2', 'src', 'tgt']]
        return df.drop_duplicates()

    def _get_control_input_df(self):
        # xcxc I'm also assuming that our columns in control_input stay static like so.
        control_input_df = pd.read_csv(os.path.join(self.data_dir, 'control_input.txt'),
                               names=['trajectory', 'input_num', 'ctr1', 'ctr2'])
        control_input_df['input_num'] = control_input_df['input_num'].astype('str')
        control_input_df['trajectory'] =control_input_df['trajectory'].astype('str')
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
            # Apologies for the hardcoding
            fn_number = str(int(fn.split('_')[0]))
            trajectory_number = str(int(fn.split('_')[2].split('.')[0]))
            if (fn_number, trajectory_number) not in filename_groupings:
                filename_groupings[(fn_number, trajectory_number)] = []
            filename_groupings[(fn_number, trajectory_number)].append(fn)
            
        # Then make a dataframe from this dictionary
        filename_df = self._get_initial_filename_dataframe(filename_groupings)
        return filename_df
    
    def _get_initial_filename_dataframe(self, filename_groupings):
        '''
        Given the filename groupings from the above, create a dataframe
        of the schema [trajectory, index, image1, image2]
        '''
        filename_df = pd.DataFrame(columns=['trajectory', 'index', 'src'])
        for k,v in filename_groupings.items():
            (index, traj) = k
            img1 = None
            if len(v) == 1:
                img1 = v[0]
            filename_df = filename_df.append({
                'trajectory': traj,
                'index': index,
                'src': img1
            }, ignore_index=True)
        filename_df['trajectory'] = filename_df['trajectory'].astype('str')
        filename_df['index'] = filename_df['index'].astype('int')
        filename_df = filename_df.dropna(subset=['src']) # Drop if any of our images is None.
        return filename_df
    
    def _get_pairwise_df(self, filename_df):
        pairwise_df = pd.DataFrame(columns=['trajectory', 'index', 'src', 'tgt'])
        trajectory_map = self._construct_trajectory_map(filename_df)
        for trajectory, goal_fn in trajectory_map.items():
            fn_subset_df = filename_df[filename_df['trajectory']==trajectory]
            pairwise_df = self._get_pairwise_combinations_for_goal(
                goal_fn, fn_subset_df, pairwise_df)
        pairwise_df['trajectory'] = pairwise_df['trajectory'].astype('str')
        pairwise_df['index'] = pairwise_df['index'].astype('str')
        return pairwise_df
    
    def _construct_trajectory_map(self, filename_df):
        '''
        Constructs a map such that
        {trajectory: goal}
        So then it's just a matter of iterating through this map.
        '''
        if len(self.goal_images) > 0:
            return self.goal_images
        trajectories = filename_df['trajectory'].unique().tolist()
        def helper(traj):
            return filename_df[filename_df['trajectory']==traj]['src'].max()
            
        goal_filenames = map(lambda t: helper(t), trajectories)
        goal_filenames = list(goal_filenames)
        return {trajectories[i]: goal_filenames[i] for i in range(len(trajectories))}
    
    def _get_pairwise_combinations_for_goal(self, goal_image, filename_df, pairwise_df):
        '''
        With filename_df, we construct the ('index', 'src', 'tgt' here), constructed by 
        '''
        num_rows, _ = filename_df.shape
        tgt_index = int(goal_image.split('_')[0]) # Get which # image we want to go up to
        
        for i in range(num_rows):
            # Get data from our current row
            ith_row = filename_df.iloc[i]
            index = int(ith_row['index'])
            # Get all the potential target images
            src_filename = ith_row['src']
            timestep = 1 # images increment by 1
            indices = list(np.arange(index, tgt_index, self.delta * timestep)) # Hardcoding in 4 because images increment by 4
            if self.delta != 1:
                indices.append(index + timestep) # And to get t+1 as well.
            tgt_rows = filename_df[filename_df['index'].astype('int').isin(indices)] # Get all the target rows
            # Then loop through our filenames and pair them together and append them to our df
            for tgt_filename in tgt_rows['src']:
                pairwise_df = pairwise_df.append({
                    'trajectory': ith_row['trajectory'],
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
    