# Neuro-Inspired Efficient Map Building via Fragmentation and Recall
Official implementation for **[Neuro-Inspired Efficient Map Building via Fragmentation and Recall (FarMap)]()**.



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
@article{hwang2023neuro,
    author = {Jaedong, Hwang and Hong, Zhang-Wei and Chen, Eric and Boopathy, Akhilan and Agrawal, Pulkit and Fiete, Ila},
    title = {Neuro-Inspired Efficient Map Building via Fragmentation and Recall},
    journal={arXiv preprint arXiv:2307.05793},
    year = {2023},
}   
```
