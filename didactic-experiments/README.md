# Law-of-the-Unconscious-Contrastive-Learner
Code for didactic experiments from the paper Law of the Unconscious Contrastive Learner: Probabilistic Alignment of Unpaired Modalities.

To run the code, first install the dependencies using `requirements.txt`
```
pip install -r requirements.txt
```

To run trials of Unconscious experiments, run `run_trial.py` with the desired command line arguments, specified in `main`.

To analyze the results of the experiments, use the provided notebook `analyze_experiments.ipynb`. The helper file `combine.py` is provided to aggregate csv files from different experiments.

Additional code for our reinforcement learning experiments can be found here: https://github.com/YongweiChe/hyperbolic-gcrl.
