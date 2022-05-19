 Understanding RL agents using generative visualisation and differentiable environment simulation
===============

This README provides instructions for how to replicate the results in our paper. 

Overview of steps:

- Train agent on procgen task
- Record dataset of real agent-environment rollouts
- Train generative model on recorded dataset of real agent-environment rollouts
- Run analyses on recorded dataset of real agent-environment rollouts
- Record dataset of simulated agent-environment rollouts from the generative model
- Run analyses on the recorded simulated rollouts. 
- Analysis of the prediction quality over time

All scripts should be run from the root dir.

First install the package locally. Navigate to the folder `train-procgen-pytorch`
and use this command in the CLI to install in editable mode:
> pip install -e .

To train the agent on coinrun:

> python train.py --exp_name [agent_training_experiment_name] --env_name coinrun --param_name hard-rec --num_levels 1000000 --distribution_mode hard --num_timesteps 200000000 --num_checkpoints 500

This will save training data and a model in a directory in
> logs/procgen/coinrun/[your_experiment_name]/

Each training run has a unique seed, so each seed gets its own directory in the 
above folder like so:
> logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]

Then to plot the training curve for that training run:

> python plot_training_csv.py --datapath="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]"

You can render your trained agent to see what its behaviour looks
like:

> python render.py --exp_name=[agent_rendering_experiment_name] --env_name="coinrun" --distribution_mode="hard" --param_name="hard-local-dev-rec" --device="cpu" --model_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth"

Assuming your agent is behaviour as you'd like it to, now we can start 
interpreting it. 

# Making recordings and training the generative model 

To begin interpretation, we need to record a bunch of agent-environment 
rollouts in order to train the generative model:

> python record.py --exp_name [recording_experiment_name] --env_name coinrun --param_name hard-rec --num_levels 1000000 --distribution_mode hard --num_checkpoints 200 --model_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth" --logdir="[path_to_rollout_data_save_dir]"

Note that ``--logdir`` should have plenty of storage space (100's of GB).

With this recorded data, we can start to train the generative model on 
agent-environment rollouts:

[comment]: <> (> python train_gen_model.py --agent_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth" --param_name=hard-rec --log_interval=10 --batch_size=28 --num_sim_steps=28 --num_initializing_steps=3 --save_interval=10000 --lr=1e-4 --env_name=coinrun --loss_scale_obs=1000.0 --loss_scale_hx=1.0 --loss_scale_reward=0.01 --loss_scale_done=0.1 --loss_scale_act_log_probs=0.00001 --loss_scale_gen_adv=0. --loss_scale_kl=1.0 --tgm_exp_name=[generative_model_training_experiment_name] --data_dir=[path_to_real_rollout_data_save_dir])
> python train_gen_model.py --agent_file=./logs/procgen/coinrun/trainhx_1Mlvls/seed_498_07-06-2021_23-26-27/model_80412672.pth --gen_mod_exp_name=dev --model_file="generative/results/rssm53_largepos_sim_penalty_extraconverterlayers/20220106_181406/model_epoch3_batch20000.pt"

That'll take a 1-4 days to train on a single GPU. Once it's trained, we'll record some agent-
environment rollouts from the model. This will enable us to compare the 
simulations to the true rollouts and will help us understand our generative 
model (which includes the agent that we want to interpret) better. This is how
we record samples from the generative model:

> python record_gen_samples.py --agent_file=./logs/procgen/coinrun/trainhx_1Mlvls/seed_498_07-06-2021_23-26-27/model_80412672.pth --gen_mod_exp_name=dev --model_file="generative/results/rssm53_largepos_sim_penalty_extraconverterlayers/20220106_181406/model_epoch3_batch20000.pt"

Now we're ready to start some analysis. 

# Analysis

The generative model is a VAE, and therefore consists of an encoder and decoder.
The decoder is the part we want to interpret because it simulates agent-
environment rollouts. It will be informative, therefore, to get a picture of
what's going on inside the latent vector of the VAE, since this is the input
to the decoder. 

## Analysis of bottleneck vector

In theory, the distribution of the VAE latent vector space is trained to be as close
as possible to a standard multivariate gaussian distribution. In practice, however, the KL 
divergence never reaches zero so the distribution of the latent vector never
becomes a perfectly Gaussian We produce PCA and and tSNE plots of the VAE
latent vectors to observe the structure of the distribution. 

> python bottleneck_vec_analysis_precompute.py
> 
> python bottleneck_vec_analysis_plotting.py

## Analysis of agent's hidden state
We'll next analyse the agent's hidden state with a few dimensionality reduction
methods. First we precompute the dimensionality reduction analyses:
> python hidden_analysis_precompute.py

with 10'000 episodes (not samples). Increase request for memory and compute time to cope with more episodes.  

which will save the analysis data in ``analysis/hx_analysis_precomp/``

Next we'll make some plots from the precomputed analyses of the agent's hidden
states:
> python hidden_analysis_plotting.py

These depict what the agent is 'thinking' during many episodes, visualised
using several different dimensionality reduction and clustering methods. 

## Analysis of environment hidden states

> python env_h_analysis_precompute.py

with 20'000 samples of len 24.  Increase request for memory and compute time to cope with more samples.  

then

> python env_h_analysis_plotting.py


[//]: # (## Analysis of SensoriMotorLoop space)

[//]: # ()
[//]: # (> python sml_analysis_precompute.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --generated_data_dir=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples)

[//]: # ()
[//]: # (## Analysis of the prediction quality over time)

[//]: # (We measure the mean squared error of each component of the generative model's loss, and see how it changes with the)

[//]: # (number of simulation steps the generative model produces. To run this experiment and output a json file with the results, run:)

[//]: # (> python loss_over_time_exp.py --exp_name demo2 --epochs 1 --batch_size 200 --agent_file=[your pth file] --device cpu --param_name hard-local-dev-rec --model_file=[your pt file])

[//]: # ()
[//]: # (Note that you may need to add arguments for the scaling factors of each loss component &#40;e.g. --loss_scale_obs=1000.0 --loss_scale_hx=1.0&#41;. To create a line plot using the data from the above experiment, run:)

[//]: # (> python analysis/plot_loss_over_time.py --presaved_data_path=generative/analysis/loss_over_time/[json output file above]")


## Calculating saliency maps


Saliency maps calculate the gradient (averaged over noised samples) of some
network quantity (e.g. the agent's value function output) with respect to inputs
or intermediate network activations.
We can thus calculate how important dimensions of the generated observations or
agent hidden states are for the value function. 

Say we wanted to generate saliency maps with respect to value and leftwards actions
for specifically the generated samples numbered 33 39 56 84. We'd use the 
following command:
> python saliency_experiments.py --agent_file=./logs/procgen/coinrun/trainhx_1Mlvls/seed_498_07-06-2021_23-26-27/model_80412672.pth --gen_mod_exp_name=dev --model_file="generative/results/rssm53_largepos_sim_penalty_extraconverterlayers/20220106_181406/model_epoch3_batch20000.pt"

[//]: # (Update text below)
If we wanted to generate saliency maps for the same quantities but combine those
samples into one sample by taking their mean latent space vector (instead of 
iterating over each sample individually), we'd add
the flag ``--combine_samples_not_iterate`` 

If we wanted to generate saliency maps for all samples from 0 to 100, we'd replace
the ``--sample_ids 33 39 56 84`` flag with ``--sample_ids 0 to 100``.

## Identifying causal stories for behaviours
After we've calculated the saliency maps, we can use them to identify the 
causal structure of the control algorithm used by the agent.

First we cluster the agent-environment dynamics. These clusters 
correspond to behaviours. 
> python analysis/combined_agent_env_hx_analysis_precompute.py

We need to summarise the IC dynamics for each behaviour. 
> python xcorr_and_xcaus_plots.py

Then we compare the magnitude and sign of the corresponding entries
in the cross-correlation and Jacobian matrices to identify where
gradients are consistent with the dynamics, both with and without
passing gradients through the environment.
> python analysis/dynamics_grads_consistency_plots.py


## Validating hypotheses by controlling the dynamics

If our hypotheses about the role of different directions in hidden-state 
space are correct, we should be able to make predictions about how the agent 
should behave when those directions are altered. 

We can record the hidden states while either swapping different directions in 
hidden-state-space or collapsing directions into the nullspace so that the agent
can't use those directions. 

We can use the `record_informinit_gen_samples.py` script to do this. 

By default, the CLI arguments for `--swap_directions_from` and 
`--swap_directions_to` are empty. If we want to swap the 10th hx direction
with the 12th hx direction and at the same time collapse the 5th hx direction
into the nullspace, we simply add the arguments
> --swap_directions_from 10 5 --swap_directions_to 12 None 

It's also advised to change the directory that the recordings get saved to in
order not to overwrite previous data from the unaltered agent hx dynamics. To do
this add something like:
> --data_save_dir=generative/recorded_validations_swapping
