# Menu
- [A Survey on Role-Oriented Network Embedding](#role-survey)
    - [Models](#models)
    - [Dataset](#dataset)
    - [Experiment](#experiment)
 
# A Survey on Role-Oriented Network Embedding
## Models 
RolX [1], RIDεRs [2], GraphWave [3], SEGK [4], struc2vec [5], struc2gauss [6], Role2vec [7], NODE2BITS [8], DRNE [9], GraLSP [10], GAS [11], RESD [12].

## Dataset
### Synthetic datasets
| Dataset  | Parameters |
| :----: | :----: | 
| Regular network |  nx.random_graphs.random_regular_graph(3, N) |
| ER network | nx.random_graphs.erdos_renyi_graph(N, 0.1) |
| Small world network | nx.random_graphs.watts_strogatz_graph(N, 3, 0.5) |
| Scale-free network | nx.random_graphs.barabasi_albert_graph(N, 3) |
### Real-world datasets
| Dataset |  Nodes | Edges | Classes |
| :----: | :----: |  :----: | :----: |
| Brazilian air-traffic network | 131 | 1,074 | 4 |
| European air-traffic network | 399 | 5,995 | 4 |
| American air-traffic network | 1,190 | 13,599 | 4 |
| Reality-call network| 6,809 | 7,697 | 3 |
| Actor co-occurrence network | 7,779 | 26,733| 4 |
| English-language film network | 27,312 | 122,706 | 4 |
| ht-wiki-talk | 446 | 758 | Bots: 24 , Admins:  0 |
| br-wiki-talk | 1,049 | 2,330 | Bots: 35 , Admins:  8 |
| cy-wiki-talk | 2,101 | 3,610 | Bots: 31 , Admins: 16 |
| oc-wiki-talk | 3,064 | 4,098 | Bots: 43 , Admins: 4 |
| eo-wiki-talk | 7,288 | 14,266 | Bots: 120 , Admins: 21 |
| gl-wiki-talk | 7,935 | 19,887 | Bots: 12 , Admins: 14 |

## Experiment
| Task |  Dataset | File & Command |  Function |
| :----: | :----: |  :----: |  :----: |
| Efficiency analysis | Regular network、ER network、Small world network、 Scale-free network、Air-traffic networks、Reality-call network、Actor co-occurrence network、English-language film network | python3 main.py| This file is used for learning the node embedding of different datasets through different methods. The run time is stored in runningtime.txt. The parameters required for the experiment are adjustable in this file.| 
| Visualization analysis | Brazilian air-traffic network | python3 brazil-flights_tsne.py | This file is used for visualization analysis of the network. |
| Classification | Air-traffic networks、Reality-call network、Actor co-occurrence network、English-language film network | python3 classification.py | This file is used to implement classification tasks, the results of which are saved in F1_micro.txt and F1_miacro.txt.The parameters required for the experiment are adjustable in this file.|
| Clustering results | Air-traffic networks、Reality-call network、Actor co-occurrence network、English-language film network | python3 clustering.py | This file is used to implement clustering tasks, the results of which are saved in NMI.txt and silhouette.txt.The parameters required for the experiment are adjustable in this file. |
| Top-k similarity search | ht-wiki-talk、br-wiki-talk、cy-wiki-talk、oc-wiki-talk、eo-wiki-talk、gl-wiki-talk | python3 top_k.py | This file is used to implement top-k similarity search, the results of which are saved in admins_acc.txt and bots_acc.txt.The parameters required for the experiment are adjustable in this file. |

### References
[1] K. Henderson, B. Gallagher, T. Eliassi-Rad, H. Tong, S. Basu, L. Akoglu, D. Koutra, C. Faloutsos, and L. Li, “Rolx: structural role extraction & mining in large graphs,” in Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2012, pp. 1231–1239.

[2] P. V. Gupte, B. Ravindran, and S. Parthasarathy, “Role discovery in graphs using global features: algorithms, applications and a novel evaluation strategy,” in 2017 IEEE 33rd International Conference on Data Engineering (ICDE), 2017, pp. 771–782.

[3] C. Donnat, M. Zitnik, D. Hallac, and J. Leskovec, “Learning structural node embeddings via diffusion wavelets,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018, pp. 1320–1329.

[4] G. Nikolentzos and M. Vazirgiannis, “Learning structural node representations using graph kernels,” IEEE Transactions on Knowledge and Data Engineering, 2019.

[5] L. F. Ribeiro, P. H. Saverese, and D. R. Figueiredo, “struc2vec: Learning node representations from structural identity,” in Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2017, pp. 385–394.

[6] Y. Pei, X. Du, J. Zhang, G. Fletcher, and M. Pechenizkiy, “struc2gauss: Structural role preserving network embedding viagaussian embedding,” Data Mining and Knowledge Discovery, vol. 34, no. 4, pp. 1072–1103, 2020.

[7] N. K. Ahmed, R. A. Rossi, J. B. Lee, T. L. Willke, R. Zhou, X. Kong, and H. Eldardiry, “Role-based graph embeddings,” IEEE Transactions on Knowledge and Data Engineering, pp. 1–1, 2020. 

[8] D. Jin, M. Heimann, R. A. Rossi, and D. Koutra, “Node2bits: Compact time-and attribute-aware node representations for user stitching,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2019, pp. 483–506. 

[9] K. Tu, P. Cui, X. Wang, P. S. Yu, and W. Zhu, “Deep recursive network embedding with regular equivalence,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018, pp. 2357–2366.

[10] Y. Jin, G. Song, and C. Shi, “Gralsp: Graph neural networks with local structural patterns,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, 2020, pp. 4361–4368.

[11] X. Guo, W. Zhang, W. Wang, Y. Yu, Y. Wang, and P. Jiao, “Roleoriented graph auto-encoder guided by structural information,” in International Conference on Database Systems for Advanced Applications, 2020, pp. 466–481.

[12] W. Zhang, X. Guo, W. Wang, Q. Tian, L. Pan, and P. Jiao, “Rolebased network embedding via structural features reconstruction with degree-regularized constraint,” Knowledge-Based Systems, p.106872, 2021.
