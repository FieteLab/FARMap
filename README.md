# Grid Cell-Inspired Fragmentation and Recall for Efficient Map Building
Official implementation for **[Grid Cell-Inspired Fragmentation and Recall for Efficient Map Building (FARMap)](https://openreview.net/forum?id=cT8oOJ6Q6F)** in TMLR 2024 *(with Featured Certification)*.
We propose a new framework for mapping based on Fragmentation-And-Recall, or FARMap, that
exploits grid cell-like map fragmentation via surprisal combined with a long-term memory to perform
efficient online map building.
Our experiments show
that FARMap reduces wall-clock time and the number of steps (actions) taken to map large spaces,
and requires a smaller online memory size relative to baselines.



<p align="center">
  <img align="middle" src="./assets/architecture.png" alt="Architecture"/>
</p>


## Run Experiments
### Setup
```
pip install -r requirements.txt
```


### (Optional) Environment Generation
```
python3 environment_generation/generator.py
```


### Exploration


```
python3 run_exp.py -wandb -env $ENV_ID
```


## Citation
```
@article{hwang2024grid,
    author = {Jaedong, Hwang and Hong, Zhang-Wei and Chen, Eric and Boopathy, Akhilan and Agrawal, Pulkit and Fiete, Ila},
    title = {Grid Cell-Inspired Fragmentation and Recall for Efficient Map Building},
    journal={TMLR},
    year = {2024},
}   
```
