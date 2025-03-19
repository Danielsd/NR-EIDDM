import numpy as np

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2=np.sqrt((nodes-node)**2)
    return np.min(dist_2)

def mean_delay(nodes):

    if(len(nodes)>0):
        dist1 = closest_node(9500,nodes)
        dist2 = closest_node(19500, nodes)
        dist3 = closest_node(29500, nodes)
        return (dist1+dist2+dist3)/3
    else:
        return 0