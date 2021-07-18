import pandas as pd
import networkx as nx
from texttable import Texttable
from gensim.models.doc2vec import TaggedDocument
import pickle as pkl

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def load_graph(graph_path):
    """
    Reading an edge list csv to an NX graph object.
    :param graph_path: Path to the edhe list csv.
    :return graph: NetworkX object.
    """
    print(graph_path)
    if "/clf/" in graph_path:
        graph = nx.from_edgelist(pd.read_csv(graph_path, encoding='utf-8', header=None,
                                             sep=' ').values.tolist())
        graph.remove_edges_from(nx.selfloop_edges(graph))
    elif "/lp/" in graph_path:
        datafile = open(graph_path,'rb')
        graphAttr = pkl.load(datafile)
        graph = graphAttr['G_rmd']
        datafile.close()
    print('Nodes:{} Edges:{}'.format(len(graph.nodes()),len(graph.edges())))
    return graph

def create_documents(features):
    """
    Created tagged documents object from a dictionary.
    :param features: Keys are document ids and values are strings of the document.
    :return docs: List of tagged documents.
    """
    docs = [TaggedDocument(words = v, tags = [str(k)]) for k, v in features.items()]
    return docs
