# DADAO
Implementation of ICML 2023 paper [DADAO: Decoupled Accelerated Decentralized Asynchronous Optimization](https://proceedings.mlr.press/v202/nabli23a.html). \
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
Simply run the main script, e.g., as follows
```bash
python main.py --optimizer_name "DADAO" --n_workers 10 --classification True --graph_type "random_geom" --t_max 200
```
In our [Examples Notebook]( https://github.com/AdelNabli/DADAO/blob/main/Examples.ipynb), we provide further examples of how to run the implemented optimizers, along with a small exploration of the datasets and graphs considered.

## Citation
```bibtex


@InProceedings{pmlr-v202-nabli23a,
  title = 	 {{DADAO}: Decoupled Accelerated Decentralized Asynchronous Optimization},
  author =       {Nabli, Adel and Oyallon, Edouard},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {25604--25626},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
}
```
