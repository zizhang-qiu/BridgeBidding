This directory contains codes for supervised learning of Bridge bidding.

The code is provided with a environment of Contract Bridge to create observation tensors in `bridge_state.py`.

# Making dataset
You can run the file `make_dataset.py` to generate a dataset of Bridge bidding.

For default, we use the dataset played by [WBridge5](http://www.wbridge5.com), which is released by the author of 
[Human-Agent Cooperation in Bridge Bidding](https://arxiv.org/abs/2011.14124). You can download the dataset [here](https://console.cloud.google.com/storage/browser/openspiel-data/bridge).



You can also create your own dataset using the encoding of OpenSpiel, in short

- The cards are represented as `rank * kNumSuits + suit`.
- The calls are represented in sequence: Pass, Dbl, RDbl, 1C, 1D, 1H, 1S, etc, starting from 52.
- Each trajectory in dataset is a sequence of actions, where first 52 actions are dealing, starting from north and going clockwise. Then the bidding and playing follows.



By running `make_dataset.py`, you will get datasets in `save_dir`, which are python `dicts` of `str, torch.Tensor.`

# Training

After making datasets, you can run `train.py` to train the network. 

The arguments are explained below:

- dataset_dir: the directory to your dataset made.
- save_dir: the directory to save trained network.
- learning_rate: the learning rate of optimizer.
- train_batch_size: the batch size for train set.
- valid_batch_size: the batch size for valid set.
- num_episodes: the number of iterations.
- eval_freq: the frequency of evaluation (how many iterations).
- device: the device to train and evaluate on.