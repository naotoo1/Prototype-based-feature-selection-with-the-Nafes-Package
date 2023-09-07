# Nafes
[Nana A. Otoo](https://github.com/naotoo1)


Nafes is an interface to of the prototype-based feature selection algorithm (under construction).


## Abstract
This paper introduces Nafes as a prototype-based feature selection package designed as a wrapper
centered on the highly interpretable and powerful Generalized Matrix Learning Vector Quantization
(GMLVQ) classification algorithm and its local variant (LGMLVQ). Nafes utilizes the learned
Relevance evaluated by the mutation validation scheme for Learning Vector quantization (LVQ)
which iteratively converges to selected features that relevantly contribute to the prototype-based
classifier decisions

[https://vixra.org/abs/2308.0112](https://vixra.org/abs/2308.0112)


The implementation requires Python 3.10 and above. The author recommends to use a virtual environment or Docker image.
The details of the implementation and results evaluation can be found in the paper.

To install the Python requirements use the following command:

 ## Requirements

```python
pip install -r requirements.txt 
```

To replicate results in the paper run the default parameters:

```python
python train.py --dataset wdbc --model gmlvq 
python train.py --dataset wdbc --model lgmlvq 
python train.py --dataset ozone --model gmlvq
python train.py --dataset ozone --model lgmlvq

```
 ## How to use

```python
usage: prototype_feature_extractor.py [-h] [--ppc PPC] [--dataset DATASET] [--model MODEL] [--bs BS] [--lr LR] [--bb_lr BB_LR] [--eval_type EVAL_TYPE] [--epochs EPOCHS] [--verbose VERBOSE]
                                      [--significance SIGNIFICANCE] [--norm_ord NORM_ORD] [--evaluation_metric EVALUATION_METRIC] [--perturbation_ratio PERTURBATION_RATIO] [--termination TERMINATION]
                                      [--perturbation_distribution PERTURBATION_DISTRIBUTION] [--optimal_search OPTIMAL_SEARCH] [--reject_option REJECT_OPTION] [--epsilon EPSILON]
                                      [--proto_init PROTO_INIT] [--omega_init OMEGA_INIT]


```
Altenatively the prototype_feature_selection algorithm is part of Nafes package which is available at and can be intalled by 

```python
pip install nafes
```

