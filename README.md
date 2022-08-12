# Adversarial Image Denoiser: A Method to Protect Visual Recommenders from Adversarial Attacks.

Install the framework following the instruction on [ELLIOT-INSTALLATION.md](ELLIOT-INSTALLATION.md)

To download the datasets check the README.md instructions in the ```/data``` folder.

The experiments can be re-run using the following list of commands:


```
python start_experiments.py --config attack_vbpr_amazon_boys_girls
```
```
python start_experiments.py --config attack_amr_amazon_boys_girls
```
```
python start_experiments.py --config attack_vbpr_amazon_men
```
```
python start_experiments.py --config attack_amr_amazon_men
```
```
python start_experiments.py --config attack_vbpr_pinterest
```
```
python start_experiments.py --config attack_amr_pinterest
```

All the configuration files are uploaded in the  ```/config_files``` folder. Note that they are customizable (e.g., gpu: 0 allows to activate the GPU 0).

The full set of results reported in the submission are available in the ```/article-results``` folder. 

We have implemented the entire pipeline integrating them in the original models of the framework. We refer to  ```/elliot/recommender/adversarial/AMR``` and ```/elliot/recommender/visual/VBPR```.
