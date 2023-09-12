# Prototype-based Feature Selection with the Nafes Package
[Nana A. Otoo](https://github.com/naotoo1)

This repository contains the code for the paper Prototype-based Feature Selection with the Nafes Package (under construction)


## Abstract
This paper introduces Nafes as a prototype-based feature selection package designed as a wrapper
centered on the highly interpretable and powerful Generalized Matrix Learning Vector Quantization
(GMLVQ) classification algorithm and its local variant (LGMLVQ). Nafes utilizes the learned
Relevance evaluated by the mutation validation scheme for Learning Vector quantization (LVQ)
which iteratively converges to selected features that relevantly contribute to the prototype-based
classifier decisions

[https://vixra.org/abs/2308.0112](https://vixra.org/abs/2308.0112)


The implementation requires Python 3.11.5 and above. The author recommends to use a virtual environment or Docker image.
The details of the implementation and results evaluation can be found in the paper.

To install the Python requirements use the following command:

```python
pip install -r requirements.txt 
```

To replicate results for WDBC in the paper run the default parameters:

```python
python train.py --dataset wdbc --model gmlvq --eval_type ho
python train.py --dataset wdbc --model gmlvq --eval_type mv
python train.py --dataset wdbc --model lgmlvq --eval_type ho --reject_options
python train.py --dataset wdbc --model lgmlvq --eval_type mv --reject_options

```

To replicate results for Ozone Layer in the paper run the default parameter:
```python
python train.py --dataset ozone --model gmlvq --eval_type ho
python train.py --dataset ozone --model gmlvq --eval_type mv
python train.py --dataset ozone --model lgmlvq --eval_type ho --reject_options
python train.py --dataset ozone --model lgmlvq --eval_type mv --reject_options

```
 

