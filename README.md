# optimized_foveation_using_rl

This is the primary repository that is being utilized for a COMP 755, Machine Learning project by Akshay Paruchuri, Chi-Jane Chen, Omar Shaban, and Chun-Hung Chao.

## Instructions

To run the main script that involves re-training of our policy agent, enter the following in a command-line context from the dirty_RL folder:

```
python main.py
```

To run the eval script that involves the evaluation of a policy agent given some checkpoint file, enter the following in a command-line context from the dirty_RL folder. Checkpoints will be stored in dirty_RL/ckpt.

```
python eval.py
```

Summarized results will be printed out in the terminal window upon completion of evaluation. Foveated images will appear in the outputs folder.