# UM-GNN

Scripts and demo for [`Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks`](https://arxiv.org/abs/2009.14455), AAAI 2021.

<img src="/files/umgnn.png" alt="UM-GNN" title="UM-GNN">

To run the demo [`demo/demo.ipynb`](./demo/demo.ipynb), follow these steps:
1. Install the `rise` module for Jupyter notebooks using pip. For conda instructions visit [`Link`](https://pypi.org/project/rise/)

```
        pip install rise
```

2. Install the `jupyter_nbextensions_configurator` using pip. For conda instructions visit [`Link`](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator)

```
        pip install jupyter_nbextensions_configurator
        jupyter nbextensions_configurator enable --user
```

3. Open `jupyter notebook` and under the Nbextensions menu, enable `Split Cells Notebook` and customize settings for `Rise` (theme: white, disable show_buttons_on_startup)

4. Invoke the demo by pressing ⌥R (For your OS check the `rise.shortcuts.slideshow` in Nbextensions)
