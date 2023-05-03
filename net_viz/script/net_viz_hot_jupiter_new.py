# %%
import networkx as nx
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

# %%

def parse_thermo_variables(fileName):
    temp = int(fileName.split("network/")[1].split("k_")[0])
    kzz = fileName.split("network/")[1].split("kzz")[1].split(".")[0]
    return temp, kzz

def load_graph_from_file(fileName, no_isolated=True):
    g = nx.read_graphml(fileName)
    if no_isolated:
        isolated = sorted([n for n in g.nodes() if g.degree(n) == 0])
        g.remove_nodes_from(isolated)
    return g

def pos_fruchterman_reingold_layout(g: object, w='logWeight'):
    iter = 700; kfactor = 120; th = 1e-16; seed = 500
    # iter = 200; kfactor = 50; th = 1e-16; seed = 1000;
    return nx.fruchterman_reingold_layout(g, iterations=iter, k=kfactor, threshold=th, seed=seed, weight=w)

def adjust_label_pos(nodePos: dict, direction: str = "down", amount: float = 0.05) -> dict:
    #if direction == "down":
    x_adjust = 0; y_adjust = - amount

    if direction == "up":
        x_adjust = 0; y_adjust = amount
    if direction == "right":
        x_adjust = amount; y_adjust = 0
    if direction == "left":
        x_adjust = - amount; y_adjust = 0

    adjusted_pos = dict()
    for n in nodePos:
        x = nodePos[n][0] + x_adjust
        y = nodePos[n][1] + y_adjust
        adjusted_pos[n] = [x, y]
    return adjusted_pos


def get_node_size_attribute(g, nodeAttr):
    return [math.pow(1.3, n[1][nodeAttr] - 82) + 50 for n in g.nodes(data=True)]


def get_edge_size_attribute(g,edgeAttr):
    return [math.pow(1.6, g[u][v][edgeAttr] - 152) - math.pow(0.1, 1.5) for u, v in g.edges()]


def renormalize_node_size_attribute(g, nodeAttr):
    min_nodeAttr = np.amin([n[1][nodeAttr] for n in g.nodes(data=True)])
    max_nodeAttr = np.amax([n[1][nodeAttr] for n in g.nodes(data=True)])
    list_nodesize = [math.pow(35, (((n[1][nodeAttr] - min_nodeAttr)/(max_nodeAttr - min_nodeAttr) + 0.3) ** 3)) for n in g.nodes(data=True)]
    const = np.sum(list_nodesize)
    #return [i / const for i in list_nodesize
    print( [(n[1][nodeAttr] - min_nodeAttr)/(max_nodeAttr - min_nodeAttr) for n in g.nodes(data=True)])
    return list_nodesize


def renormalize_edge_size_attribute(g, edgeAttr):
    min_edgeAttr = np.amin([g[u][v][edgeAttr] for u, v in g.edges()])
    max_edgeAttr = np.amax([g[u][v][edgeAttr] for u, v in g.edges()])
    list_edgesize = [math.pow(35,(((g[u][v][edgeAttr] - min_edgeAttr)/(max_edgeAttr - min_edgeAttr)) ** 20)) - 2 for u, v in g.edges()]
    #list_edgesize = [(g[u][v][edgeAttr] - min_edgeAttr) / (max_edgeAttr - min_edgeAttr) ** 3 for u, v in g.edges()]
    const = np.sum(list_edgesize)
    #return [i / const for i in list_edgesize]
    # print([(g[u][v][edgeAttr] - min_edgeAttr)/(max_edgeAttr - min_edgeAttr)  +  0.05 for u, v in g.edges()])
    return list_edgesize


def draw_single_net_with_attribute(fileName, nodesize='logAbundance', edgesize='logWeight'):
    t, k = parse_thermo_variables(fileName)

    g = load_graph_from_file(fileName)

    pos_node = pos_fruchterman_reingold_layout(g)
    pos_label = pos_node

    list_nsize = get_node_size_attribute(g, nodeAttr=nodesize)
    list_esize = get_edge_size_attribute(g, edgeAttr=edgesize)

    plt.figure(figsize=(5,5))
    plt.subplot(1,1,1)
    plt.axis('off')

    nx.draw_networkx_nodes(g, pos_node, node_size=list_nsize)
    nx.draw_networkx_labels(g, pos_label, font_size=5)
    nx.draw_networkx_edges(g, pos_node, width=list_esize, arrows=False)


def draw_multiple_nets_with_attribute(list_fileName, pathOutput, nodesize='logWDegree', edgesize='logWeight', show=True):

    # To fix the positions of nodes and labels
    g = load_graph_from_file(list_fileName[-1])
    pos_node = pos_fruchterman_reingold_layout(g)
    #pos_label = adjust_label_pos(pos_node, amount=0.1)
    pos_label = pos_node

    plt.figure(figsize=(30, 6.5))
    sub_index = 1

    for fileName in list_fileName:

        t, k = parse_thermo_variables(fileName)

        g = load_graph_from_file(fileName)

        list_nsize = get_node_size_attribute(g, nodeAttr=nodesize)
        list_esize = get_edge_size_attribute(g, edgeAttr=edgesize)

        plt.subplot(1, len(list_fileName), sub_index)
        plt.axis('off')
        #plt.title("kzz = %s, %dK\n"%(k,t))

        nx.draw_networkx_nodes(g, pos_node, node_size=list_nsize,
                               node_color='#aeccdc')
        nx.draw_networkx_labels(g, pos_label, font_size=11)
        nx.draw_networkx_edges(g, pos_node, alpha=0.9, width=list_esize,
                               edge_color="#767676", arrows=False)

        sub_index += 1

    plt.savefig(pathOutput, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()


def draw_multiple_renormalized_nets_with_attribute(list_fileName, pathOutput, nodesize='logWDegree', edgesize='logWeight', show=True):

    # To fix the positions of nodes and labels
    g = load_graph_from_file(list_fileName[-1])
    pos_node = pos_fruchterman_reingold_layout(g)
    #pos_label = adjust_label_pos(pos_node, amount=0.1)
    pos_label = pos_node

    plt.figure(figsize=(30, 6.5))
    sub_index = 1

    for fileName in list_fileName:

        t, k = parse_thermo_variables(fileName)

        g = load_graph_from_file(fileName)

        list_nsize = renormalize_node_size_attribute(g, nodeAttr=nodesize)
        list_esize = renormalize_edge_size_attribute(g, edgeAttr=edgesize)

        plt.subplot(1, len(list_fileName), sub_index)
        plt.axis('off')
        #plt.title("kzz = %s, %dK\n"%(k,t))

        nx.draw_networkx_nodes(g, pos_node, node_size=list_nsize,
                               node_color='#aeccdc')
        nx.draw_networkx_labels(g, pos_label, font_size=11)
        nx.draw_networkx_edges(g, pos_node, alpha=0.9, width=list_esize,
                               edge_color="#767676", arrows=False)

        sub_index += 1

    plt.savefig(pathOutput, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()

#%%
dir_data = "/Users/hkim78/work/2020-hotJupiter/data/network"  # dir for network file
dir_viz = "/Users/hkim78/work/2020-hotJupiter/net_viz/viz"  # dir for visualizations

list_temp = [500, 1000, 1500, 2000]
list_file_name = list()
# for temp in list_temp:
#
#     file_name = os.path.join(dir_data, "%dk_1met_kzz0.graphml"%temp)
#     list_file_name.append(file_name)
#
# draw_multiple_nets_with_attribute(list_file_name, os.path.join(dir_viz, "kzz0_graphs.pdf"))
#
# list_file_name2 = list()
# for temp in list_temp:
#
#     file_name = os.path.join(dir_data, "%dk_1met_kzz1e10.graphml"%temp)
#     list_file_name2.append(file_name)
#
# draw_multiple_nets_with_attribute(list_file_name2, os.path.join(dir_viz, "kzz1e10_graphs.pdf"))

for temp in list_temp:

    file_name = os.path.join(dir_data, "%dk_1met_kzz0.graphml"%temp)
    list_file_name.append(file_name)

draw_multiple_renormalized_nets_with_attribute(list_file_name, os.path.join(dir_viz, "renormalized_kzz0_graphs.pdf"))

list_file_name2 = list()
for temp in list_temp:

    file_name = os.path.join(dir_data, "%dk_1met_kzz1e10.graphml"%temp)
    list_file_name2.append(file_name)

draw_multiple_renormalized_nets_with_attribute(list_file_name2, os.path.join(dir_viz, "renormalized_kzz1e10_graphs.pdf"))
