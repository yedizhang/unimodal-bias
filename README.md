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

Underparameterized and Overparameterized two-layer early/late fusion linear networks.

```bash
python main.py --mode early_fusion --data multi --plot_weight --plot_Eg --in_dim 100 --dataset_size 70 --epoch 1200 --noise 0.5
python main.py --mode early_fusion --data multi --plot_weight --plot_Eg --in_dim 100 --dataset_size 700 --epoch 1200 --noise 0.5
python main.py --mode late_fusion --data multi --plot_weight --plot_Eg --in_dim 100 --dataset_size 70 --epoch 1200 --noise 0.5
python main.py --mode late_fusion --data multi --plot_weight --plot_Eg --in_dim 100 --dataset_size 700 --epoch 1200 --noise 0.5
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
python main.py --mode deep_fusion --init 0.1 --plot_weight --epoch 100000 --sweep rho_sweep --repeat 5
python main.py --mode deep_fusion --init 0.1 --plot_weight --epoch 100000 --sweep ratio_sweep --repeat 5
python main.py --mode deep_fusion --plot_weight --epoch 100000 --sweep init_sweep --repeat 5
```


## Citation

```
@InProceedings{yedi24unimodal,
  title = 	 {Understanding Unimodal Bias in Multimodal Deep Linear Networks},
  author =       {Zhang, Yedi and Latham, Peter E. and Saxe, Andrew M},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {59100--59125},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhang24aa/zhang24aa.pdf},
  url = 	 {https://proceedings.mlr.press/v235/zhang24aa.html}
}
```
