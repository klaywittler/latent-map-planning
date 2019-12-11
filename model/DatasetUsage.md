### How to use the DataSets
Author: Ed Y (hyoo).

#### `CarlaDataset` usage instructions
 * `data_dir`: Where the data is (should have _out directory with the images and the input .txt file)
 * `load_as_grayscale`: Boolean to decide whether to load as grayscale or not
 * `transform`: Your pytorch transforms

#### `VelocityPredictionCarlaDataSet` usage instructions
 * `data_dir`: Where the data is (should have _out directory with the images and the input .txt file)
 * `goal_images`: Optional Parameter, default {}. A map of {trajectory: goal image}. Trajectory and goal image should be strings. An example would be 
    ```
    {'1': 'trajectory1_goal.png', '2': 'trajectory2_goal.png'}
    ```
    If no parameter is given, we simply take the last images of each trajectory as our goal.
 * `delta`: Optional Parameter, default 100. An integer that defines what frequency to pair up our images. If an image is at t0, we will get as data points `[(t0,t1), (t0, t0 + delta), (t0, t0 + 2 * delta)....]`.  
 **Warning**: Do not make this delta too small! Even with 100, we get ~4k data points. 
 * `load_as_grayscale`: Boolean to decide whether to load as grayscale or not
 * `transform`: Your pytorch transforms
