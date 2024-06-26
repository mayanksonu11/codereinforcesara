"""
    This code allows to determine if an  NSLR (Network Slice Request)
    can be granted or not based on the available resources from the substrate
    several vnfs can be accepted in the same node
    
    23 oct 19:
    se adiciona funcion reduce_nslr_graph para reducir tamano del nslr graph

    04 nov 19:
    se adiciona fruncionalidad para poder mapear vnfs en nodos de otro tipo
    en caso de que los nodos del tipo necesario no tienen recursos disponibles;
    De esta forma, aceptar nslrs de un caso de uso tendra mayor influencia sobre
    la aceptacion de nslrs de otro caso de uso
"""

import copy
import networkx as nx
import nsl_request
from operator import itemgetter, attrgetter, methodcaller 
# import graph_generator as ggen

ranked_nodes_cpu = []
nsl_graph_red = {} #reduced nsl graph


def delay_node(percent_util):# to change the function 
    if percent_util<1:
        delay= 0.001/(1-percent_util)
    else:
        delay=1000
    return delay # output in ms

def nsl_placement(nslr, substrate, node_to_nslr):
    # initialization of variables
    global ranked_nodes_cpu
    curr_id = nslr.id
    profit_nodes = 0
    profit_links = 0
    centralized_vnfs = []
    # local_vnfs = []
    edge_vnfs = [] 
    delay = 0    
    current_delay = 0
    delta_delay = 0 
    
    #builds a reduced version of the nsl_graph 
    reduce_nslr_graph(nslr) 
    vnodes = nslr.nsl_graph_reduced["vnodes"]    
    # assert(nsl_graph_red == nslr.nsl_graph_reduced)
    # other required parameters like node_list, ranked_nodes,
    calculate_resource_potential(substrate,"cpu")
    nodes = copy.deepcopy(substrate.graph["nodes"]) # copy to temporarily work with
    ranked_nodes_cpu = sort_nodes(nodes,"node_potential") #ranked list of nodes by potential considering cpu and conections

    # Flags
    rejected = False
    flag = False # to know if a vnode has not been mapped to nodes of the same type
    iter_rejected = False
    ################### vnfs admission #################
    already = [] #list of nodes that already hold a vnode
    already_vnf = dict() # list of VNF already places with it's current delay

    for v in vnodes:
        if rejected:
            break
        for n in ranked_nodes_cpu:  
            iter_rejected = False
            #if sufficient resource and node of the same type and unused node, vnode accepted  
            current_delay  = delay_node(1-n["cpu"]/n["init_cpu"])
            next_percent_utilization= 1-((n["cpu"]-v["cpu"])/n["init_cpu"])
            expected_delay = delay_node(next_percent_utilization)
            delta_delay = expected_delay - current_delay

            if(expected_delay > nslr.delay_budget):
                rejected = True
                break

            # check if the addition of this delta delay affects other existing users on the node
            if n["id"] in node_to_nslr.keys():
                for _nslr in node_to_nslr[n["id"]]:
                    if(_nslr.current_delay + delta_delay > _nslr.delay_budget):
                        # print("Delay violation of existing users due to percent_util:",percent_utilization, " demand:",v["cpu"])
                        iter_rejected = True
                        break
                if iter_rejected:
                    continue                
           
            
            # find the best VNF with corresponding delay
            _delay = delay

            # Below code caters to the fact that there could be multiple nodes and we shouldn't add delays of backup nodes as only one of them will be used during service
            if v["function"] not in already_vnf:
                _delay = _delay + delay_node(next_percent_utilization) 
            else:
                if already_vnf[v["function"]] > delay_node(next_percent_utilization):
                    _delay = _delay - already_vnf[v["function"]] + delay_node(next_percent_utilization)
            assert(_delay > 0)

            if _delay <= nslr.delay_budget and v["cpu"] <= n["cpu"] and v["type"] == n["type"] and n["id"] not in already: #
                #mapping:
                #mapping of user ids to the nodes that they are connected using nodes id are the index
                v["mapped_to"] = n["id"] 
                delay = _delay

                # This code is continuation of previous comment
                if v["function"] not in already_vnf:
                    already_vnf[v["function"]] = delay_node(next_percent_utilization)
                else:
                    already_vnf[v["function"]] = min(delay_node(next_percent_utilization), already_vnf[v["function"]])

                already.append(n["id"])
                break
            else: # recurso insuficiente, vnode rejected               
                if ranked_nodes_cpu.index(n) == len(ranked_nodes_cpu)-1: #slice rejection only when no node has enough resources
                    rejected = True    
                    break   
    ################### ------------- #################                
    if(len(already) != len(vnodes)):
        rejected = True
    ################## vlinks admission #################
                
    if not rejected:
        # print(nslr.nsl_graph_reduced)
        rejected = analyze_links_dijkstra(nslr.nsl_graph_reduced,substrate)
    
    if not rejected:
        assert(delay > 0)
        nslr.current_delay = delay
        nslr.set_nsl_graph_reduced(nslr.nsl_graph_reduced)
        # update the delay of all the users on the current node in case of successful admission of user
        for node in already:
            if node in node_to_nslr.keys():
                for _nslr in node_to_nslr[node]:
                    _nslr.current_delay += delta_delay

    assert(curr_id == nslr.id)
    return rejected, delay 

def sort_nodes(node_list,sortby):       
    sorted_list = sorted(node_list, key=itemgetter(sortby), reverse=True)#sorted list    
    return sorted_list

def calculate_resource_potential(substrate,resource_type):
    '''
        potential or importance of the node
        the potential of a node to embed vnfs in terms of cpu, str, and bw
        local_rsc_capacity  =  node resource * sumatoria de bw de todos los links adyacentes al nodo
        degree_centrality = degree/(n-1)
        node potential = local_rsc_capacity + degree_centrality
        
    '''
    nodes = substrate.graph["nodes"]
    links = substrate.graph["links"]
    for i in range(len(nodes)):
        bw_sum = 0
 
        for l in range(len(links)):            
            if links[l]["source"] == i or links[l]["target"] == i: #links conectados a nodo i
                bw_sum += links[l]["bw"] #sum of outgoing links bw 
        
        local_rsc_capacity = nodes[i].get(resource_type) * bw_sum
        # nodes[i]["node_potential"] = (local_rsc_capacity/10000) + (nodes[i]["degree_centrality"]*5)
        # nodes[i]["node_potential"] = local_rsc_capacity #+ (nodes[i]["degree_centrality"]*5)
        nodes[i]["node_potential"] = nodes[i]["cpu"]
        #print("+++",local_rsc_capacity)
        #print("+++",nodes[i]["degree_centrality"])


def reduce_nslr_graph(nslr):
    '''
        Procesamiento previo del nslr graph para reducir su tamano
        las vnfs que se puede intanciar en un mismo nodo fisico se agrupan como un unico nodo virtual
    '''
    centralized_vnfs = []
    # local_vnfs = []
    edge_vnfs = []

    #1. Agrupar vnfs por tipo de nodo:
    for vnf in nslr.nsl_graph["vnfs"]: #recorrer vnfs       
        if vnf["type"] == 0:
            centralized_vnfs.append(vnf)
        # elif vnf["type"] == 1:
        #     local_vnfs.append(vnf)
        else: 
            edge_vnfs.append(vnf) 

    #2. Sort vnfs by backup:
    centralized_vnfs = sorted(centralized_vnfs, key=itemgetter("backup"))
    # local_vnfs = sorted(local_vnfs, key=itemgetter("backup"))
    edge_vnfs = sorted(edge_vnfs, key=itemgetter("backup"))

    #3. Agrupar vnfs que se instanciaran en un mismo nodo fisico:
    nsl_graph_red["vnodes"] = []
    group_vnfs(centralized_vnfs,0)
    # group_vnfs(local_vnfs,1)
    group_vnfs(edge_vnfs,1)

    #4. Crear nuevos vlinks considerando los virtual nodes creados
    nsl_graph_red["vlinks"] = []
    new_vlinks(nsl_graph_red,nslr.nsl_graph)

    #5. graph resultante es agregado al nslr
    nslr.set_nsl_graph_reduced(nsl_graph_red) 
    # nslr["nsl_graph_red"] = nsl_graph_red
    # print("**",nsl_graph_red)

def group_vnfs(vnfs_list,node_type):
    '''
        creates different vnodes and adds them to the reduced graph
    '''
    vnode = {}
    vnode["vnfs"] = [] #ids de las vnfs que conforman este vnode
    vnode["type"] = node_type
    
    if len(nsl_graph_red["vnodes"]) == 0:
        cont = 0
    else:
        cont = len(nsl_graph_red["vnodes"])
    
    for i in range(len(vnfs_list)):
        # print(vnfs_list[i])
        if i==0:
            vnode["cpu"] = vnfs_list[i]["cpu"]
            vnode["function"] = vnfs_list[i]["function"]
            vnode["vnfs"].append(vnfs_list[i]["id"])
            if i == len(vnfs_list)-1:
                vnode["id"]=cont
                nsl_graph_red["vnodes"].append(vnode.copy())
                cont += 1

        elif vnfs_list[i]["backup"]==vnfs_list[i-1]["backup"]:
            vnode["cpu"] += vnfs_list[i]["cpu"]
            vnode["vnfs"].append(vnfs_list[i]["id"])
            if i == len(vnfs_list)-1:
                vnode["id"]=cont
                nsl_graph_red["vnodes"].append(vnode.copy())
                cont += 1

        else:            
            vnode["id"]=cont 
            nsl_graph_red["vnodes"].append(vnode.copy())
            cont += 1
            vnode["cpu"] = vnfs_list[i]["cpu"]
            vnode["function"] = vnfs_list[i]["function"]
            vnode["vnfs"] = []
            vnode["vnfs"].append(vnfs_list[i]["id"])
            if i == len(vnfs_list)-1:
                vnode["id"]=cont
                nsl_graph_red["vnodes"].append(vnode.copy())
                cont += 1

    # return nsl_graph_red

def new_vlinks(nsl_graph_red, nsl_graph):
    vnfs = nsl_graph["vnfs"]
    vnodes = nsl_graph_red["vnodes"]
    new_vlink = {}
    for vlink in nsl_graph["vlinks"]:     
        source = next(vnf for vnf in vnfs if vnf["id"]==vlink["source"])
        target = next(vnf for vnf in vnfs if vnf["id"]==vlink["target"])
        #si src y target son del mismo tipo y son del mismo backup, entonces no haga nada (vlink no se considera)
        if source["type"] == target["type"] and source["backup"] == target["backup"]:
            pass
        else: #caso contrario vlink si se considera, ya que el par de vnfs estan en vnodes distintos
            new_src = next(vnode for vnode in vnodes if source["id"] in vnode["vnfs"])
            new_tgt = next(vnode for vnode in vnodes if target["id"] in vnode["vnfs"]) 
            new_vlink = {"source":new_src["id"],"target":new_tgt["id"],"bw":vlink["bw"]}
            nsl_graph_red["vlinks"].append(new_vlink)


def analyze_links(nsl_graph,substrate):
    '''
    Make the decision to accept or reject based on shortest path
    Find the shortest path with enough bw in each link to instantiate
    a v_link. Max number of hops allowed is 5
    If there is no path with hops <= 5 and enough bw, the nslr is rejected 
    '''

    G = nx.node_link_graph(substrate.graph)#graph format
    links = copy.deepcopy(substrate.graph["links"])#copia para trabajar temporalmente con ella
    reject = False
    max_hops = 5
    vlinks = nsl_graph["vlinks"]
    vnfs = nsl_graph["vnodes"]
    for vlink in vlinks:
        substrate_src = next(vnf["mapped_to"] for vnf in vnfs if vnf["id"] == vlink["source"]) 
        substrate_dst = next(vnf["mapped_to"] for vnf in vnfs if vnf["id"] == vlink["target"])
        # print("\n***vlink:",vlink)
        # que hacer con las vnfs que se instancian en el mismo nodo? cobrar por vlink? cuanto?
        paths = nx.all_simple_paths(G,source=substrate_src,target=substrate_dst)
        path_list = [p for p in paths]
        path_list.sort(key=len)
        for path in path_list:
            #verificar si todos los links en el path tienen recurso suficiente
            enough = True
            # print("*PATH:",path)
            if len(path) >= max_hops:
                reject = True
                # print("hops number is 5 or higher")
                break
            else:    
                for l in range(len(path)-1):

                    link = next(lk for lk in links if ( (lk["source"]==path[l] and lk["target"]==path[l+1]) or (lk["source"]==path[l+1] and lk["target"]==path[l]) ) )
                    # print("*",path[l],path[l+1])
                    # print("link:",link["bw"])
                    if vlink["bw"] <= link["bw"]: #hay suficiente bw                        
                        link["bw"] -= vlink["bw"] #resource is updated
                        # enough bw                       
                    else:# not enough bw
                        enough = False

                if enough:
                    # print("MAPEAR")
                    vlink["mapped_to"] = path#si hubo bw suficiente en cada link del path, se mapea
                    break
                elif enough == False and path_list.index(path) == len(path_list)-1:
                        reject = True              
                
        if reject:
            break

    return reject

def analyze_links_dijkstra(nsl_graph,substrate):
    '''
    Make the decision to accept or reject based on shortest path
    Find the shortest path with enough bw in each link to instantiate
    a v_link. Max number of hops allowed is 5
    If there is no path with hops <= 5 and enough bw, the nslr is rejected 
    '''

    G = nx.node_link_graph(substrate.graph)#graph format
    links = copy.deepcopy(substrate.graph["links"])#copia para trabajar temporalmente con ella
    reject = False
    max_hops = 5
    vlinks = nsl_graph["vlinks"]
    vnfs = nsl_graph["vnodes"]
    # print("In Analyze links:",vnfs)
    for vlink in vlinks:
        substrate_src = next(vnf["mapped_to"] for vnf in vnfs if vnf["id"] == vlink["source"]) 
        substrate_dst = next(vnf["mapped_to"] for vnf in vnfs if vnf["id"] == vlink["target"])
        # print("\n***vlink:",vlink)
        # que hacer con las vnfs que se instancian en el mismo nodo? cobrar por vlink? cuanto?
        path_ = nx.shortest_path(G,source=substrate_src,target=substrate_dst,weight="bw")
        paths = [path_]
        path_list = [p for p in paths]
        path_list.sort(key=len)
        for path in path_list:
            #verificar si todos los links en el path tienen recurso suficiente
            enough = True
            # print("*PATH:",path)
            if len(path) >= max_hops:
                reject = True
                # print("hops number is 5 or higher")
                break
            else:    
                for l in range(len(path)-1):

                    link = next(lk for lk in links if ( (lk["source"]==path[l] and lk["target"]==path[l+1]) or (lk["source"]==path[l+1] and lk["target"]==path[l]) ) )
                    # print("*",path[l],path[l+1])
                    # print("link:",link["bw"])
                    if vlink["bw"] <= link["bw"]: #hay suficiente bw                        
                        link["bw"] -= vlink["bw"] #resource is updated
                        # enough bw                       
                    else:# not enough bw
                        enough = False

                if enough:
                    # print("MAPEAR")
                    vlink["mapped_to"] = path#si hubo bw suficiente en cada link del path, se mapea
                    break
                elif enough == False and path_list.index(path) == len(path_list)-1:
                        reject = True              
                
        if reject:
            break

    return reject


# substrate = ggen.ba_graph(str(1))
# print(substrate)
# decision = nsl_placement(NSLR, substrate)





