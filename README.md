# *Graphs Go With The Flow:* Flow Matching For Graph Generation
In this report, we focus on three novel adaptations of flow matching to discrete data inspired by
variational inference, inherent simplex geometry, and topics from information geometry, namely, [1], [2], and [3, 4]. We reproduce the models and
test their performance on the graph generation task across unified settings.

## Experiments

### Swiss Roll on Simplex

To reproduce this toy experiment from [4], use our adaptation of the [original](https://github.com/ccr-cheng/statistical-flow-matching/blob/main/swissroll.ipynb) `swissroll.ipynb` notebook, where we additionally included our implementation of CatFlow [1] with an MLP backbone as in [4].

### QM9: Categorical Generation

Following [1], we use the DiGress [5] graph backbone (and its [original implementation](https://github.com/cvignac/DiGress/tree/main)) for this series of experiments.
#### Dirichlet FM [1]:
**TBD @ Yiting**
#### CatFlow [1]:
The model's code is located in the `src/catflow` directory. 

* To run hyperparameter tuning, launch:
```
python tune_catflow.py
```
* To train the model, launch: 
```
python train_catflow.py 0
```
* To generate graphs, perform:
```
python train_catflow.py 1
```
In case you want to obtain the corresponding metrics right away, use the `generate.py` script.
#### Fisher-FM [3, 4]:
**TBD @ Felix**

### QM9: Mixed Continuous and Categorical Generation

As in [3], we perform this series of experiments on the graph net from FlowMol (and its [original implementation](https://github.com/Dunni3/FlowMol)).
#### Dirichlet FM [1]:
We reproduce the results based on the implementation in the [FlowMol repo](https://github.com/Dunni3/FlowMol).
#### CatFlow [1]:
* After following the setup of [FlowMol repo](https://github.com/Dunni3/FlowMol), to train the CatFlow model, run:
```
python train.py --config=configs/qm9_ctmc.yaml
```
* To generate and score the inferred graphs, run:
```
python test.py --model_dir=flowmol/trained_models/qm9_simplexflow --n_mols=100 --n_timesteps=250 --output_file=brand_new_molecules.sdf
```
#### Fisher-FM [3, 4]:
We use the original implementation of the Fisher-Flow model and thus refer to the [official repo](https://github.com/olsdavis/fisher-flow).

## Data: QM9 

To set up the QM9 dataset, follow the instructions in the [FlowMol repo](https://github.com/Dunni3/FlowMol).

## Metrics

For the categorical generation, we adapt the implementation of the corresponding metrics from the [MoFlow repo](https://github.com/calvin-zcx/moflow). In the mixed continuous and categorical generation, we use implementations from the [FlowMol repo](https://github.com/Dunni3/FlowMol).


## References
* [1]: [Variational Flow Matching for Graph Generation](https://arxiv.org/abs/2406.04843v1) (Eijkelboom et al., NeurIPS 2024).
* [2]: [Dirichlet Flow Matching with Applications to DNA Sequence Design](https://arxiv.org/abs/2402.05841) (Stark et al., ICML 2024).
* [3]: [Fisher Flow Matching for Generative Modeling over Discrete Data](https://arxiv.org/abs/2405.14664) (Davis et al., NeurIPS 2024).
* [4]: [Categorical Flow Matching on Statistical Manifolds](https://arxiv.org/abs/2405.16441) (Cheng et al., NeurIPS 2024).
* [5]: [Digress: Discrete denoising diffusion for graph generation](https://arxiv.org/abs/2209.14734) (Vignac et al., ICLR 2023).
