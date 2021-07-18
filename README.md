# Menu
- [A Survey on Role-Oriented Network Embedding](#role-survey)
    - [Models](#models)
    - [Dataset](#dataset)
    - [Experiment](#experiment)
 
# A Survey on Role-Oriented Network Embedding
## Models 
RolX、RIDεRs、GraphWave、SEGK、struc2vec、struc2gauss、Role2vec、NODE2BITS、DRNE、GraLSP、GAS、RESD

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