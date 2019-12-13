Chris Hsu [chsu8@seas],
Klayton Wittler [kwittler@seas],
Hohun Yoo [hyoo@seas]

Latent Map Planning
ESE 546- final project

CVAE.ipynb - CNN-VAE colab notebook
siameseCVAE.ipynb - siamese CNN-VAE colab notebook trained on 2 camera view data
PredictsiameseCVAE.ipynb - siamese CNN-VAE colab notebook trained on t and t+1 data
InterpolateSiameseCVAE.ipynb - Given a checkpoint of a model, interpolate between 2 images
plotElls.ipynb - Plot saved pickle files that recorded losses
velocityModel.ipynb - Given a checkpoint, train a classifier to predict control inputs

carla/ - contains all necessary scripts to run in carla 0.9.5
The distribution package is not included for sake of submission
Scripts inside carla/agents were used for data collection
latent_agent.py - testing agent for the control prediction model
