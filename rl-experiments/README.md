### Simplified Installation Instructions 
```
module load anaconda3/2023.3
conda create -n env python=3.10
conda activate env 
pip install torch hypll wandb graphviz ipython matplotlib
```
if you haven't used wandb, export WANDB_MODE=offline when running slurm scripts


### How to Run  Experiments
Install requirements:
```
pip install -r requirements.txt
```

`train.py` run contrastive reinforcement learning on a simple maze environment.

`train_streets.py` runs unconscious contrastive reinforcement using (state, state-action) pairs and (state, language) pairs.
