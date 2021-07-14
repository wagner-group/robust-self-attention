# Robust Self Attention

This repo is based on the [pycls](https://github.com/facebookresearch/pycls) project.
If you already have pycls installed you will need to create a new python environment to install this codebase.
Most original pycls functionality should remain intact however, this has not been thoroughly tested.


## Setup

### Code

Clone the repository:

```
git clone https://github.com/wagner-group/robust-self-attention
```

Install PyTorch from [pytorch.org](https://pytorch.org).

Install additional dependencies:

```
pip install -r requirements.txt
```

Set up modules:

```
python setup.py develop --user
```

### Data

Please see [`DATA.md`](./docs/DATA.md) for setting up the ImageNet dataset.
The ImageNet-100 subset is specified by setting the `MODEL.NUM_CLASSES` config option to 100.

## Evaluation
```
python tools/test_net.py --cfg configs/patchadv/eval_resnet.yaml ADV.VAL_PATCH_SIZE 10
```

## Training

Download ImageNet-100 weights from [here](https://drive.google.com/drive/folders/1_2Od8rMSFqUE9dQn5Nqta6129GK_lqGm?usp=sharing) and adversarially finetune model with:

```
python tools/train_net.py --cfg configs/patchadv/train_resnet_adv.yaml
```

Additional evaluation and training configs used in the paper are available in the configs folder.

## Changes

This repo includes several changes to the original pycls functionality:

- support for adversarial training and evaluation
- WandB logging
- load RGB images by default instead of BGR
- more flexible checkpoint loading
- slurm submission scripts using [submitit](https://github.com/facebookresearch/submitit)

as well as many new options accessible through the config system- see diff history for `config.py` for full list of new options.
