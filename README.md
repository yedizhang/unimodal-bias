# Understanding Unimodal Bias in Multimodal Deep Linear Networks

[Project Website](https://yedizhang.github.io/unimodal-bias)  |  [Paper](https://arxiv.org/abs/2312.00935)

## Setup

Python 3 dependencies:

- pytorch
- numpy
- scipy
- argparse
- matplotlib

## Usage

The minimal example: two-layer early/late fusion linear networks.

```bash
python main.py --mode early_fusion --plot_weight
python main.py --mode late_fusion --plot_weight
```

Sweep experiments.

Two-layer late fusion linear network sweep.
```bash
python main.py --mode late_fusion --plot_weight --epoch 40000 --sweep toy_sweep --repeat 5
python main.py --mode late_fusion --plot_weight --data multi --lr 0.005 --epoch 20000 --sweep rand_sweep --repeat 50
```

Deep multimodal linear network sweep.
```bash
python main.py --mode deep_fusion --init 0.1 --plot_weight --sweep depth_single --epoch 7000
python main.py --mode deep_fusion --init 0.07 --plot_weight --epoch 100000 --sweep rho_sweep --repeat 5
python main.py --mode deep_fusion --init 0.07 --plot_weight --epoch 100000 --sweep ratio_sweep --repeat 5
python main.py --mode deep_fusion --plot_weight --epoch 100000 --sweep init_sweep --repeat 5
```


## Citation

```
@misc{yedi2023unimodal,
      title={A Theory of Unimodal Bias in Multimodal Learning}, 
      author={Yedi Zhang and Peter E. Latham and Andrew Saxe},
      year={2023},
      eprint={2312.00935},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
