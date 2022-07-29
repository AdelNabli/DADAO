# DADAO
Implementation of [DADAO: Decoupled Accelerated Decentralized Asynchronous Optimization for Time-Varying Gossips]( https://hal.archives-ouvertes.fr/hal-03737694/document ). \
To compare our work to other decentralized optimizers, we also implemented [MSDA](https://arxiv.org/pdf/1702.08704.pdf), [ADOM+](https://openreview.net/attachment?id=L8-54wkift&name=supplementary_material) _(with and without Multi-Consensus)_ and the optimizer described in the [Continuized Framework](https://arxiv.org/pdf/2106.07644.pdf).

## Requirements
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pytorch](https://pytorch.org/)
* [networkx](https://networkx.github.io/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pandas](https://pandas.pydata.org/)
* [tqdm](https://tqdm.github.io/)

## Usages
Simply run the main scipt, e.g., as follows
```bash
python main.py --optimizer_name 'DADAO' --n_workers 10 --classification True --graph_type 'random_geom' --t_max 200
```
We provide examples of how to run the implemented optimizers in our Examples Notebook.
