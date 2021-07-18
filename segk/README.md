## Learning Structural Node Representations using Graph Kernels
Code for the paper [Learning Structural Node Representations using Graph Kernels](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8869809).

### Requirements
Code is written in Python 3.6 and requires:
* grakel 0.1a6
* scikit-learn 0.21

### Basic Usage
To run *SEGK* on Barbell graph, execute the following command from the project home directory:<br/>
``python segk.py --path-to-edgelist datasets/barbell.edgelist --path-to-output-file embeddings/barbell.txt``

#### Hyperparameters
The following three hyperparameters can be specified:
* radius: the maximum radius of the neighborhood subgraphs
* dim: the dimensionality of the generated embeddings
* kernel: the employed graph kernel (either the shortest path kernel or the weisfeiler lehman kernel)

#### Input
The supported input format is an edgelist where the endpoints are separated by the space character:

    node1_id node2_id
        
#### Output
The output file contains *n* lines (where *n* is the number of nodes of the graph) as follows:

    node_id dim1 dim2 ... dimd
where dim1, ... , dimd is the *d*-dimensional representation learned by *SEGK*.

### Experiments
In the project home directory, there are four jupyter notebooks that reproduce the experiments presented in the paper.

### Cite
Please cite our paper if you use this code:
```
@article{nikolentzos2019learning,
  title={Learning Structural Node Representations using Graph Kernels},
  author={Nikolentzos, Giannis and Vazirgiannis, Michalis},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019}
}
```

-----------

Provided for academic use only
