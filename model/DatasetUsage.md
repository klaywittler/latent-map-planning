### How to use the DataSets
Ed Y.

CarlaDataset usage instructions
    data_dir: Where the data is (should have _out directory with the images and the input .txt file)
    load_as_grayscale: Boolean to decide whether to load as grayscale or not
    transform=your transforms

VelocityPredictionCarlaDataSet usage instructions
    data_dir: Where the data is (should have _out directory with the images and the input .txt file)
    goal_images: [OPTIONAL PARAMETER] A map of {trajectory: goal image}. Trajectory and goal image should be strings. 
    delta: At what frequency to pair up our images
    load_as_grayscale: Boolean to decide whether to load as grayscale or not
    transform: your transforms

A warning on VelocityPredictionCarlaDataSet: DO NOT HAVE DELTA TO BE TOO SMALL!!! Even with delta=100, you get ~4k data points.
