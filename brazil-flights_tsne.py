import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

color_pool = ['#2ECC71', '#2980B9', '#48C9B0', '#F7DC6F', '#8E44AD', '#EC7063', '#F39C12', '#E74C3C',
              '#5DADE2', '#1ABC9C', '#27AE60', '#F39C12', '#EDBB99', '#F5B7B1', '#FAD7A0']
#'RolX','RIDεRs','GraphWave','SEGK','struc2vec','struc2gauss','role2vec','node2bits','DRNE','GraLSP'
methods=['RolX','RIDeRs-S','GraphWave','segk','struc2vec','struc2gauss','role2vec','node2bits','DRNE','GraLSP','GAS','RESD']
x=['RolX','RIDεRs','GraphWave','SEGK','struc2vec','struc2gauss','Role2vec','NODE2BITS','DRNE','GraLSP','GAS','RESD']
dataset = 'brazil-flights'
label = pd.read_csv('dataset/clf/{}.lbl'.format(dataset), header=None, sep=' ')
label = label.sort_values(0).values[:, 1]
tsne = TSNE(n_components=2)
fig = plt.figure(figsize=(24, 12))
for count in range (1,len(methods)+1):#对每个方法
    method = methods[count-1]
    ax = fig.add_subplot(2,6,count)
    ax.set_xlabel('{}'.format(x[count-1]),fontsize = 32)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    try:
        embed=pd.read_csv('embed/{}/clf/{}_128.emb'.format(method,dataset))
    except:
        embed=pd.read_csv('embed/{}/clf/{}.emb'.format(method,dataset))
    embed=embed.drop(['id'],axis=1).values
    print(embed.shape)
    print(methods)
    if embed.shape[1]>2:
        embed=tsne.fit_transform(embed)
    colors = []
    for each in label:
        colors.append(color_pool[each % len(color_pool)+2])
    plt.scatter(embed[:,0],embed[:,1],color=colors,s=100)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
plt.savefig('{}_tsne.pdf'.format(dataset),bbox_inches='tight')
plt.show()