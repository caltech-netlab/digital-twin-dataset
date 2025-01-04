# Third-party import
import os
import copy
import random
from copy import deepcopy
from dateutil import parser
import networkx as nx
import numpy as np
from numpy.linalg import inv
from shapely import wkt
import re
import matplotlib.pyplot as plt

# First-party import
import sys
import pathlib
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]) + '/utils')
import utils
import math_utils
from global_param import JOIN_CHAR, DIR_CHAR, FILE_PATHS, F


AGGREGATION_GROUP_TYPES = [
    "BuildingAggregation",
    "Building",
    "Vault",
    "Pad",
    "Plant",
    "Substation",
    "Switchgear",
    "SwitchArray",
]


"""Helper functions below"""


def to_edge_list(edges, data=False):
    """
    :param edges: dict, key: edge name, value: edge attributes
    :param data: bool, whether to include edge attributes
    :return: list of tuple of str (src, tgt) where src/tgt are source/target node names
    """
    if data:
        return {(e["source"], e["target"]): name for name, e in edges.items()}
    else:
        return [(e["source"], e["target"]) for name, e in edges.items()]


def to_edge_dict(edges, flip=False):
    """
    :param edges: dict, key: edge name, value: edge attributes
    :param flip: bool, whether to flip value and keys
    :return: dict, key: edge name, value: tuple of str (src, tgt)
        where src/tgt are source/target node names. If flip is True,
        values become keys and keys become values.
    """
    return (
        {(e["source"], e["target"]): name for name, e in edges.items()}
        if flip
        else {name: (e["source"], e["target"]) for name, e in edges.items()}
    )


def check_duplicate_edge(edges):
    edge_set = set()
    for edge in edges.values():
        if (edge["source"], edge["target"]) in edge_set:
            return True
        else:
            edge_set.add((edge["source"], edge["target"]))
    return False


def find_meter(
    graph_element, decorators, element_children, preference=("EgaugeMeter",)
):
    """
    Given element name, check if it is metered, and return the meter name if yes.
    To accomodate multiple meters measuring the same element, a list is returned.
    :param graph_element: dict, either a node or an edge
    :param decorators: dict, see net2graph
    :param element_children: dict, load from element_inheritance.json file
    :param preference: str, meter element type, preferred meter
    :return: list of str, meter element name, or None
    """
    meters_name_type = []
    if not "decorators" in graph_element:
        return meters_name_type
    for decorator_name in graph_element["decorators"]:
        decorator_type = decorators[decorator_name]["element_type"]
        if decorator_type in element_children["Meter"]:
            meters_name_type.append((decorator_name, decorator_type))
    meters_name_type.sort(key=lambda element: utils.list_index(preference, element[1]))
    return [name_type[0] for name_type in meters_name_type]


def replace_node(edge_list, node_name, new_node_name):
    """
    Replace the node in edges with a new node, searching by node_name
    :param edge_list: see net2graph
    :param node_name: str
    :param new_node_name: str
    :return: dict of dict, modified edge_list
    """
    for edge in edge_list.values():
        edge["source"] = (
            new_node_name if edge["source"] == node_name else edge["source"]
        )
        edge["target"] = (
            new_node_name if edge["target"] == node_name else edge["target"]
        )
    return edge_list


def remove_nodes_with_no_edge(nodes, edges):
    """
    Given edges, remove any node that is not connected to an edge.
    :param nodes: list of dict, see net2graph
    :param edges: list of dict, see net2graph
    :return: nodes, updated
    """
    connected_nodes = set()
    for name, edge in edges.items():
        connected_nodes.add(edge["source"])
        connected_nodes.add(edge["target"])
    to_remove = set(nodes.keys()) - connected_nodes
    for name in to_remove:
        nodes.pop(name)
    return nodes


def in_group(element, net_data, mode, group_types=AGGREGATION_GROUP_TYPES):
    """
    :param element: dict, element dictionary
    :param net_data: dict, output of load_net_files
    :param mode: str, one of {'exclusive', 'any'}
        - 'exclusive': element is considered in a group, if it belongs to one and only one group
        - 'any': element is considered in a group, if it belongs to at least one group
    :param group_types: list of str, element_types to consider
    :return: bool, whether an element is considered to be within a single group
    """
    # If the 'groups' attribute doesn't even exist, then the element is not in group
    if not ("groups" in element):
        return False
    # If the 'groups' attribute has exactly one element in a certain group, then the element is in group
    num_groups_belongs_to = sum(
        [
            utils.in_list(element["groups"], net_data[group_type])
            for group_type in group_types
            if group_type in net_data
        ]
    )
    if (mode == "exclusive") and (num_groups_belongs_to == 1):
        return True
    if (mode == "any") and (num_groups_belongs_to > 0):
        return True
    # If the 'groups' attribute has zero or more than one element in groups,
    # then the element is not in group
    return False


def get_group(element, net_data, mode, group_types=AGGREGATION_GROUP_TYPES):
    """
    :param element: dict, element dictionary
    :param net_data: dict, output of load_net_files
    :param mode: str, one of {'exclusive', 'any'}, see in_group function
        - 'exclusive': element can belong to atmost one group
        - 'any': element can belong to multiple groups
    :param group_types: list of str, element_types to consider
    :return: a tuple of
        - group type (e.g. Building, Vault)
        - group element name (e.g. noyesLab)
    """
    group_type_list, group_list = [], []
    for group in element["groups"]:
        for group_type in group_types:
            if (group_type in net_data) and (group in net_data[group_type]):
                group_type_list.append(group_type)
                group_list.append(group)
    if mode == "exclusive":
        if len(group_list) == 0:
            return None, None
        elif len(group_list) == 1:
            return group_type_list[0], group_list[0]
        else:
            raise ValueError(
                f"Element can only belong to maximum of 1 group in mode='exclusive'.\nElement: {element}"
            )
    else:
        return group_type_list, group_list


def get_group_type(group, net_data, element_children):
    """
    :param group: str, group name
    :param net_data: dict, output of load_net_files
    :param element_children: dict, load from element_inheritance.json file
    :return: str, group type name
    """
    for group_type in element_children["PhysicalGrouping"]:
        if (group_type in net_data) and (group in net_data[group_type]):
            return group_type
    raise ValueError(f"{group} does not exist. Check network files for completion.")


def find_elements_by_type(element_types, net_data):
    """
    Find all elements of a specific type in network files (e.g. find all Loads).
    :param net_data: dict, output of load_net_files
    :param element_types: list of str, element types (e.g. ["Load", "Inverter"])
    :return: list of dict (element dictionaries)
    """
    elements = []
    for et in element_types:
        if et in net_data:
            elements += list(net_data[et].keys())
    return elements


def find_element(name, net_data, element_types=[], raise_err=False, print_warning=True):
    """
    Given element name, find its type and the element dictionary.
    :param net_data: dict, output of load_net_files
    :param name: element name, not graph element name
    :param raise_err: bool
    :param print_warning: bool
    :return: a tuple of
        - str, element type
        - dict, element dictionary
    """
    if DIR_CHAR in name:
        raise ValueError(
            f"Pass in physical element name, not graph element name: {name}"
        )
    element_types = element_types or net_data.keys()
    for element_type in element_types:
        if name in net_data[element_type]:
            return element_type, net_data[element_type][name]
    # Element not found
    if raise_err:
        raise RuntimeError(f"Element {name} does not exist.")
    else:
        if print_warning:
            print(f"[ERROR] Element {name} does not exist.")
        return None, None


def opposite_node(edge, node):
    """
    Given an edge and one of its nodes, find its opposite node
    :param edge: dict, element dictionary
    :param node: str, node name
    :return: str, node name
    """
    if edge["target"] == node:
        return edge["source"]
    elif edge["source"] == node:
        return edge["target"]
    else:
        raise ValueError(f"Edge {edge} is not connected to node {node}")


def connected_edges(node_name, edges, directed=False):
    """
    :param node_name: str, graph element name (not element name)
    :param edges: dict, list of edges, see net2graph
    :param directed: bool, whether to distinguish incoming and outgoing edges
    :return: tuple of 2 dicts, k: edge name, v: edge dict
    """
    incoming, outgoing = {}, {}
    for edge_name, edge in edges.items():
        if edge["source"] == node_name:
            outgoing[edge_name] = edge
        if edge["target"] == node_name:
            incoming[edge_name] = edge
    if directed:
        return incoming, outgoing
    else:
        incoming.update(outgoing)
        return incoming


def edge_direction(edge, node_name):
    """
    :param edge: dict, see net2graph
    :param node_name: str, node name
    :return: int, +1 if edge points into the node and -1 if edge points out of the node.
    """
    if edge["source"] == node_name:
        return -1
    elif edge["target"] == node_name:
        return +1
    else:
        raise ValueError(f"Node {node_name} is not connected to edge {edge}")


def get_combined_node_attributes(node_name, nodes, net_data, attribute, check_equal=True):
    if type(nodes[node_name]) is list:
        attr = [get_element(n, net_data)[attribute] for n in nodes[node_name]]
        if check_equal:
            assert len(set(attr)) == 1, f"Attributes {attr} not equal."
            return attr[0]
        return attr
    else:
        res = get_element(nodes[node_name], net_data)[attribute]
        return res if check_equal else [res]


def shortest_path(G, source=None, target=None):
    """Wrapper for networkx shortest_path, handles NetworkXNoPath exception"""
    try:
        return nx.shortest_path_length(G, source=source, target=target)
    except nx.NetworkXNoPath:
        return float("inf")


def longest_path(G, root_nodes=None):
    """
    Find the longest path between each of the root node to any other nodes.
    :param G: networkx.DiGraph
    :param root_nodes: list of str
    :return: dictionary of dictionary,
        outer key = str, root node name
        inner key = str, node name
        value = float, distance
    """
    root_nodes = root_nodes if root_nodes else [n for n, d in G.in_degree() if d == 0]
    topo_order = list(nx.topological_sort(G))
    dist_dict = {}
    for root in root_nodes:
        dist = dict.fromkeys(G.nodes, -float("inf"))
        dist[root] = 0
        for n in topo_order:
            for s in G.successors(n):
                if dist[s] < dist[n] + 1:
                    dist[s] = dist[n] + 1
        dist_dict[root] = dist
    return dist_dict


def depth(G, root_nodes=None, return_rootnode=False):
    """
    Find the depth of each node in graph. Root nodes are at depth 0.
    If a node is connected to more than one root nodes, its depth is taken as
    the maximum of distances to each of the connected root nodes.
    :param G: networkx.DiGraph
    :param root_nodes: list of str
    :param return_rootnode: bool, whether to return another dictionary
    :return:
        depth_dict: dict of dict. key = str, node name. value = int, depth. Reverse if flip_depth_dict.
        root_dict: dict, key = node name, value = root node name
    """
    root_nodes = root_nodes if root_nodes else [n for n, d in G.in_degree() if d == 0]
    dist_dict = utils.flip_nested_dict(longest_path(G, root_nodes=root_nodes))
    depth_dict, root_dict = {}, {}
    for node in dist_dict:
        depth_dict[node] = max(dist_dict[node].values())
        if return_rootnode:
            root_dict[node] = max(dist_dict[node], key=dist_dict[node].get)
    for node in root_nodes:
        depth_dict[node] = 0
        if return_rootnode:
            root_dict[node] = node
    # Check every depth value is filled (no gaps, from zero to maximum depth)
    assert set(depth_dict.values()) == set(range(max(depth_dict.values()) + 1)), set(
        depth_dict.values()
    )
    return (depth_dict, root_dict) if return_rootnode else depth_dict


"""API-level functions below"""


def read_net_files(net_files, name_as_keys=True):
    """
    Loads json network files into dictionaries in memory.
    e.g.
    net_data = {
        "Bus": {"bus_a": {"name": "bus_a", ...}, "bus_b": {"name": "bus_b", ...}, ...},
        "Line": {"line_a": {"name": "line_a", ...}, "line_b": {"name": "line_b", ...}, ...},
        ...
    }
    :param net_files: list of str, paths to files
    :param name_as_keys: bool, if True, all elements of the same type will be in a dictionary,
        with their name as keys; if False, elements of the same type will be in a list (same as json files),
        this is deprecated. Please set name_as_keys to true.
    :return:
        - if name_as_keys is True:
            dict of dict of dict, element type -> element name -> element dictionary
        - if name_as_keys is False:
            dict of list of dict, element type -> list of element dictionaries
    """
    net_data_all = {}
    # For each network file
    for net_file in net_files:
        net_data = (
            net_file
            if type(net_file) is dict
            else utils.load_json(net_file, raise_err=True)
        )
        for element_type in net_data:
            utils.insert_dict_list(net_data_all, [element_type], net_data[element_type])
    # Turn list of elements into dict of elements
    if name_as_keys:
        return {
            element_type: utils.list2dict(net_data_all[element_type], "name")
            for element_type in net_data_all
        }
    else:
        return net_data_all


def traverse_dict_list_populate(data, t, timeseries_dir):
    if type(data) is dict:
        for k, v in data.items():
            if type(v) in (dict, list):
                data[k] = traverse_dict_list_populate(v, t, timeseries_dir)
            elif (type(v) is str) and v.startswith("file:"):
                header = np.loadtxt(
                    os.path.join(timeseries_dir, v[5:]),
                    delimiter=",",
                    dtype=str,
                    max_rows=1,
                )
                struct_arr = np.loadtxt(
                    os.path.join(timeseries_dir, v[5:]),
                    delimiter=",",
                    dtype=str,
                    skiprows=1,
                )
                t_col = struct_arr.reshape(-1, 2)[:, 0].astype("datetime64[us]")
                v_col = struct_arr.reshape(-1, 2)[:, 1].astype(header[1])
                idx = utils.np_searchsorted(
                    t_col, t, mode="nearest_before", inclusive=True, safe_clip=False
                )
                assert (
                    idx >= 0
                ), f"Parameter value undefined for time {t} in file {v[5:]}."
                data[k] = v_col[idx]
    elif type(data) is list:
        for i, v in enumerate(data):
            if type(v) in (dict, list):
                data[i] = traverse_dict_list_populate(v, t, timeseries_dir)
            elif (type(v) is str) and v.startswith("file:"):
                df = utils.read_ts(os.path.join(timeseries_dir, v[5:]))
                idx = utils.np_searchsorted(
                    df["t"], t, mode="nearest_before", inclusive=True, safe_clip=False
                )
                assert (
                    idx >= 0
                ), f"Parameter value undefined for time {t} in file {v[5:]}."
                data[i] = df["v"][idx]
    else:
        raise ValueError(f"Unsupported input argument type: {type(data)}")
    return data


# Updated (minor)
def load_net_files(
    net_files, t=None, timeseries_dir=FILE_PATHS["net_params"], name_as_keys=True
):
    """
    For our new time-dependent network topology data, wrap read_net_files with load_net_files.
    If t is not supplied, behavior is the same as the original time-independent version.
    Note: Significant performance improvement can be achieved by
        1. Caching network files and parameter timeseries to avoid reading from disk
        2. Caching the keys to the parameters with time-varying values, to avoid network data traversal
    :param t: np.datetime64 or str
    :param name_as_keys: bool, if True, all elements of the same type will be in a dictionary,
        with their name as keys; if False, elements of the same type will be in a list (same as json files),
        this is deprecated. Please set name_as_keys to true.
    :param net_files: list of str, paths to files (if t is not specified) or folders (if t is specified)
    :param timeseries_dir: str, path to folder containing timeseries of network element parameters
    :return: dict, see read_net_files
    """
    t = parser.parse(t) if type(t) is str else t
    if (not t) or all([f.endswith(".json") for f in net_files]):
        return read_net_files(net_files, name_as_keys=name_as_keys)
    else:
        # Step 1: load the appropriate .json netowrk file based on time t
        net_files_t = []
        for net_file_folder in net_files:
            files = sorted(
                [str(f) for f in os.listdir(net_file_folder) if f.endswith(".json")]
            )
            available_t = np.array([parser.parse(f[:-5]) for f in files])
            i = utils.np_searchsorted(
                available_t, t, mode="nearest_before", inclusive=True, safe_clip=False
            )
            assert (
                i >= 0
            ), f"No network topology available for time {t} among files: {os.listdir(net_file_folder)}"
            net_files_t.append(os.path.join(net_file_folder, files[i]))
        net_data = read_net_files(net_files_t, name_as_keys=name_as_keys)
        # Step 2: populate any parameter fields that are time-series
        traverse_dict_list_populate(net_data, t, timeseries_dir)
        return net_data


def get_element(graph_element, net_data):
    return net_data[graph_element["element_type"]][graph_element["element_name"]]


def get_metadata(metadata_path=FILE_PATHS["net_metadata"]):
    netfile_changes = utils.read_ts(os.path.join(metadata_path, "netfile_changes.csv"))[
        0
    ]["t"]
    parameter_changes = utils.read_ts(
        os.path.join(metadata_path, "parameter_changes.csv")
    )[0]["t"]
    topology_changes = np.unique(np.concatenate([netfile_changes, parameter_changes]))
    return netfile_changes, parameter_changes, topology_changes


def net2graph(
    net_data,
    element_parents,
    fix_implicit=True,
    connected_edges=False,
    print_warning=True,
):
    """
    Graph structure:
        nodes: Bus
        edges: PowerTransfer
        decorators: Injections, Meters
    Node and edge names are the same as element's 'name', except
        elements with multiple tbus'es (below), or when nodes are combined (see graph2electrical_graph).
    Elements that are currently ignored:
        - Control, Compute, Communication (Phase 3)
    What is an 'element dictionary'?
        They are what's generated for each element from the .json network files.
    For elements with multiple tbuses (e.g. multi-position switch):
        - an edge is created for each tbus
        - each edge is a copy of the element, with the following modification:
        - only one tbus is kept, which is the one corresponding to that edge
            i.e. element['tbus'] = [{'name': '<bus_name>', 'status': '<status'}]
        - the name of the edge is: element['name'] + DIR_CHAR + element['target']
    :param net_data: dict, output of load_net_files
    :param element_parents: str or path, from element_inheritance.json file
    :param fix_implicit: bool, whether to add in implicitly defined elements
    :param connected_edges: bool, for each node in ndoes, whether to also store
        the names of edges connected to it.
    :param print_warning: bool
    :return:
        - nodes, dict, equivalent to a node list where dictionary keys are node names
            and corresponding values are node attributes, which are
            - 'element_name', e.g. 'bus_1234'
            - 'element_type', e.g. 'Bus'
            - 'decorators', e.g. ['meter_a', 'load_2']
        - edges, dict, equivalent to an edge list where dictionary keys are edge names
            and corresponding values are edge attributes.
            - 'element_name', e.g. 'sw_2'
            - 'element_type', e.g. 'Switch'
            - 'source', e.g. 'node_a'
            - 'target', e.g. 'node_b'
            - 'decorators', e.g. ['meter_a']
        - decorators, dict, same structure as nodes and edges, with attributes
            - 'element_name', e.g. 'egauge_1'
            - 'element_type', e.g. 'BMSMeter' or 'Load'
            - 'decorated_elements', dict k:v = element name not graph name:element type
                e.g. {'bus_1234': 'Bus'}
    """
    # Output dictionaries
    nodes, edges, decorators = {}, {}, {}
    for element_type in net_data:
        for name, element in net_data[element_type].items():
            try:
                graph_element = {
                    "element_name": name,
                    "element_type": element_type,
                }
                # Buses are nodes
                if "Bus" in element_parents[element_type]:
                    graph_element.update({"decorators": []})
                    nodes[name] = graph_element
                # PowerTransfer elements are edges
                elif "PowerTransfer" in element_parents[element_type]:
                    graph_element.update({"decorators": []})
                    graph_element.update({"source": element["fbus"]})
                    if len(element["tbus"]) == 1:
                        graph_element.update({"target": element["tbus"][0]["name"]})
                        edges[name] = graph_element
                    else:
                        for t in range(len(element["tbus"])):
                            e = deepcopy(graph_element)
                            e.update({"target": element["tbus"][t]["name"]})
                            edges[name + DIR_CHAR + e["target"]] = e
                elif "ACInjection" in element_parents[element_type]:
                    graph_element.update(
                        {"decorated_elements": {element["bus"]: "Bus"}}
                    )
                    decorators[name] = graph_element
                elif "Meter" in element_parents[element_type]:
                    # Insert a new key 'decorated_elements', a dict with k:v = element name:element type
                    decorated_elements = (
                        {
                            reg["element"].split(".")[0]: find_element(
                                reg["element"].split(".")[0],
                                net_data,
                                print_warning=print_warning,
                            )[0]
                            for reg in element["registers"]
                            if "element" in reg
                        }
                        if "registers" in element
                        else {}
                    )
                    graph_element.update({"decorated_elements": decorated_elements})
                    decorators[name] = graph_element
                else:
                    pass
            except Exception as e:
                print(element)
                raise e
    # Add implicit nodes
    if fix_implicit:
        for name, edge in edges.items():
            if not edge["source"] in nodes:
                nodes[edge["source"]] = {
                    "element_name": edge["source"],
                    "element_type": "Bus",
                    "decorators": [],
                }
                utils.insert_dict(
                    net_data, ["Bus", edge["source"]], {"name": edge["source"]}
                )
                if print_warning:
                    print(
                        f"[WARNING] Added implicit node {edge['source']} referenced by edge {name}."
                    )
            if not edge["target"] in nodes:
                nodes[edge["target"]] = {
                    "element_name": edge["target"],
                    "element_type": "Bus",
                    "decorators": [],
                }
                utils.insert_dict(
                    net_data, ["Bus", edge["target"]], {"name": edge["target"]}
                )
                if print_warning:
                    print(
                        f"[WARNING] Added implicit node {edge['target']} referenced by edge {name}."
                    )
        for decorator_name, decorator in decorators.items():
            for decorated_name, decorated_type in decorator[
                "decorated_elements"
            ].items():
                if not utils.dict_exists(net_data, [decorated_type, decorated_name]):
                    if not (decorated_type is None):
                        if "Bus" in element_parents[decorated_type]:
                            utils.insert_dict(
                                net_data,
                                ["Bus", decorated_name],
                                {"name": decorated_name},
                            )
                            nodes[decorated_name] = {
                                "element_name": decorated_name,
                                "element_type": "Bus",
                                "decorators": [],
                            }
                        elif "PowerTransfer" in element_parents[decorated_type]:
                            utils.insert_dict(
                                net_data,
                                ["PowerTransfer", decorated_name],
                                {"name": decorated_name},
                            )
                            edges[decorated_name] = {
                                "element_name": decorated_name,
                                "element_type": "PowerTransfer",
                                "decorators": [],
                            }
                    if print_warning:
                        print(
                            f"[WARNING] Implicit element {decorated_name} referenced by {decorator_name}"
                        )

    # Store decorator information in nodes and edges (2-way association)
    for name, decorator in decorators.items():
        for decorated_name, decorated_type in decorator["decorated_elements"].items():
            if decorated_type is None:
                continue
            elif "Bus" in element_parents[decorated_type]:
                utils.insert_dict_list(nodes, [decorated_name, "decorators"], name)
            elif "PowerTransfer" in element_parents[decorated_type]:
                utils.insert_dict_list(edges, [decorated_name, "decorators"], name)
            elif "ACInjection" in element_parents[decorated_type]:
                utils.insert_dict_list(decorators, [decorated_name, "decorators"], name)
            else:
                raise ValueError(f"Unimplemented for type: {decorated_type}")
    # Store edge connectivity in nodes (2-way association)
    if connected_edges:
        for name, edge in edges.items():
            if edge["source"] in nodes:
                utils.insert_dict_list(nodes, [edge["source"], "edges"], name)
            if edge["target"] in nodes:
                utils.insert_dict_list(nodes, [edge["target"], "edges"], name)
        for node in nodes.values():
            if "edges" not in node:
                node["edges"] = []

    return nodes, edges, decorators


# This is updated
def graph2electrical_graph(
    nodes, edges, decorators, net_data, element_parents, prune_nodes=False
):
    """
    Turns the physical graph from net2graph into electrical graph.
    Changes to the physical graph:
        - Node list now contains 2 types of nodes
            1) normal nodes, which are element dictionaries
            2) combined nodes, which are list of element dictionaries
        - For combined nodes, their node name is all the nodes' names joined by JOIN_CHAR
        - Zero and infinite impedance edges are removed
    :param nodes: see net2graph
    :param edges: see net2graph
    :param decorators: see net2graph
    :param net_data: dict, output of load_net_files
    :param element_parents: str or path, from element_inheritance.json file
    :param prune_nodes: bool, whether to remove nodes that have no edges
    :return: nodes, edges, see net2graph
    """
    nodes, edges = copy.deepcopy(nodes), copy.deepcopy(edges)
    for name in list(edges.keys()):
        edge, edge_element = edges[name], get_element(edges[name], net_data)
        # Rule 1) If switch status is open, then remove the edge
        tbuses_status = utils.list2dict(edge_element["tbus"], "name")
        tbus_status = [
            tbuses_status[tgt]
            for tgt in edge["target"].split(JOIN_CHAR)
            if tgt in tbuses_status
        ]
        if all(
            [("status" in tbus) and (tbus["status"] == "NO") for tbus in tbus_status]
        ):
            edges.pop(name)
        # Rule 2) For zero-impedance elements (OpenClose with status NC, or
        # Lines without line rating or line_rating=None), remove the edge and collapse 2 nodes
        elif ("OpenClose" in element_parents[edge["element_type"]]) \
            or (
                ("Line" in element_parents[edge["element_type"]])
                and (("line_rating" not in edge_element)
                    or (edge_element["line_rating"] in ("None", None))
                    or ("length" not in edge_element)
                    or (edge_element["length"] == 0)
                )
            ):
            edges.pop(name)
            # Handle edge connecting a node to itself
            if edge["source"] == edge["target"]:
                continue
            src = nodes.pop(edge["source"])
            tgt = nodes.pop(edge["target"])
            src = src if type(src) is list else [src]
            tgt = tgt if type(tgt) is list else [tgt]
            # The new, combined node is a list of element dictionaries
            combined_node = src + tgt
            combined_node_name = edge["source"] + JOIN_CHAR + edge["target"]
            names_combined = edge["source"].split(JOIN_CHAR) + edge["target"].split(JOIN_CHAR)
            combined_node_name = JOIN_CHAR.join(sorted(names_combined))
            nodes[combined_node_name] = combined_node
            # Discussion: this can be sped-up if we store connectivity info in nodes
            # i.e. add key 'edges' in a node's element dictionary. Then, we can just update
            # the connected edges to src and tgt instead of iterating through all edges
            replace_node(edges, edge["source"], combined_node_name)
            replace_node(edges, edge["target"], combined_node_name)
    nodes = remove_nodes_with_no_edge(nodes, edges) if prune_nodes else nodes
    return nodes, edges, decorators


def count_elements(net_data):
    res = {}
    for type, l in net_data.items():
        res[type] = len(l)
    return res


"""Admittance Y matrix generation"""


def get_transformer_parameters(element):
    """Network element not edge element."""
    try:
        V_primary = element['kv'][0][0] * 1000
        V_secondary = element['kv'][0][1] * 1000
        VA_rating = element['kva'][0] * 1000
        percent_z = element['percent_z'][0] / 100
        connection = element['winding_connectivity']
        y_shunt = element['shunt_admittance'][0] if 'shunt_admittance' in element else 1e-7
        conn_type = element['transformer_type']
    except KeyError as e:
        print(f"Element {element['name']} has missing key(s).")
        raise e
    return V_primary, V_secondary, VA_rating, percent_z, connection, conn_type, y_shunt


# Updated
def _transformer_Y_matrix(
    V_primary, V_secondary, VA_rating, percent_z, connection, conn_type, 
    y_shunt=1e-7, returned_matrix='Y', flip_tbus_current_direction=True
):
    """
    Given transformer specs, output a 8x8 admittance matrix of a 3-phase transformer.
    Optionally, return the transmission matrix T instead.
    Reference: Coppo, M., Bignucolo, F., & Turri, R. (2017). Generalised transformer modelling
    for power flow calculation in multi-phase unbalanced networks. IET Generation, Transmission
    & Distribution, 11(15), 3843–3852. https://doi.org/10.1049/iet-gtd.2016.2080

    :param V_primary: float, unit: V, external line-to-line voltage rating of the primary side
    :param V_secondary: float, unit: V, external line-to-line voltage rating of the secondary side
    :param VA_rating: float, unit: VA, total VA power rating of the transformer, all phases combined
    :param percent_z: float, in range(0, 1) exclusive, percent Z rating (series impedance) of the transformer
    :param y_shunt: float, unit: Ohm^-1, tranformer shunt admittance
    :param connection: list of tuples of 2 str, e.g. for a delta-wye transformer,
        connection = [
            ["AB", "NB"],
            ["BC", "NC"],
            ["CA", "NA"]
        ]
    :param conn_type: type of transformer, e.g. "3PDYG", "3PDY", "3PDD"
        Specifies whether it is a 3-phase Delta-Wye, Delta-Delta, Wye-Wye, etc.
    :param returned_matrix: str, one of {'Y', 'T', 'T_inv', "component Y"}, whether to return the
        - admittance matrix Y: [I1, I2] = Y @ [V1, V2]
        - transmission matrix T: [V1, I1] = T @ [V2, I2]
        - transmission matrix T inverse: [V2, I2] = T_inv @ [V1, I1]
        - series admittance, shunt admittance, turns ratio. Each is a list of size 3.
        Note that 1 is the primary side and 2 is the secondary side.
    :param flip_tbus_current_direction: bool. The positive current direction is defined
        as INTO the transformer for both primary and secondary sides by convention.
        If flip_tbus_current_direction is true, then positive current direction is
        INTO the primary side and OUT OF the secondary side, i.e. flipped.
    :return: np.ndarray, float, 8x8 matrix. The I, V vectors are A, B, C, N.
    """
    A = np.array(
        [
            [1, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 1, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # 1) Build primitive admittance matrix Y_P
    y_series_pu = (VA_rating / 3) / percent_z
    # y1a, y1b, y1c, y2a, y2b, y2c, y0a, y0b, y0c
    admittances = [y_series_pu * 2 for _ in range(6)] + [y_shunt for _ in range(3)]
    Y_P = np.diag(admittances)

    # 2) Build winding admittance matrix Y_W
    # Connect the admittances with admittance-winding incidence matrix A
    # This gives the IV relationship I = Y_W0 @ V within the unitary voltage system.
    Y_W0 = A.transpose() @ Y_P @ A
    # Remove node 0 with Kron reduction
    Y_W0_11 = Y_W0[0:6, 0:6]
    Y_W0_12 = Y_W0[0:6, 6:9]
    Y_W0_21 = Y_W0[6:9, 0:6]
    Y_W0_22 = Y_W0[6:9, 6:9]
    Y_W = Y_W0_11 - Y_W0_12 @ inv(Y_W0_22) @ Y_W0_21

    # 3) Apply turns ratio matrix M
    # Connection will determine whether the sqrt(3) terms will be there. We only need winding voltages here.
    if conn_type[:4] == "3PYD":
        E_wc_pri = V_primary / np.sqrt(3)
        E_wc_sec = V_secondary
    elif conn_type[:4] == "3PDY":
        E_wc_pri = V_primary
        E_wc_sec = V_secondary / np.sqrt(3)
    elif conn_type[:4] == "3PYY":
        E_wc_pri = V_primary
        E_wc_sec = V_secondary
    else:
        raise RuntimeError(f"Not yet implemented transformer type: {conn_type}")
    E = np.array([E_wc_pri, E_wc_pri, E_wc_pri, E_wc_sec, E_wc_sec, E_wc_sec])
    M = np.diag(1 / E)
    Y_M = M @ Y_W @ M

    if returned_matrix == "component Y":
        a_turns_ratio = np.ones(3) * (E_wc_pri / E_wc_sec)
        return np.ones(3) * y_series_pu / E_wc_pri ** 2, np.ones(3) * y_shunt, a_turns_ratio, connection

    # 4) Finally, apply the port incidence matrix C to connect windings to external ports
    C = np.zeros((6, 8))
    port_idx = "ABCN"
    for i, conn in enumerate(connection):
        C[i, port_idx.index(conn[0][0])] = 1
        C[i, port_idx.index(conn[0][1])] = -1
        C[3+i, 4+port_idx.index(conn[1][0])] = 1
        C[3+i, 4+port_idx.index(conn[1][1])] = -1
    Y = C.transpose() @ Y_M @ C
    if flip_tbus_current_direction:
        Y = np.diag([1, 1, 1, 1, -1, -1, -1, -1]) @ Y

    # 5) Optionally, compute the transmission matrix T from admittance matrix Y. This is derived by Yiheng.
    if returned_matrix == 'Y':
        return Y
    elif returned_matrix == 'T':
        Y11 = Y[0:4, 0:4]
        Y12 = Y[0:4, 4:8]
        Y21 = Y[4:8, 0:4]
        Y22 = Y[4:8, 4:8]
        try:
            Y21_inv = np.linalg.inv(Y21)
        except:
            raise RuntimeError("Y_21 is not invertible. In general, the transmission matrix does not exist (e.g. for Delta-Wye transformers).")
        T11 = - Y21_inv @ Y22
        T12 = Y21_inv
        T21 = Y12 - Y11 @ Y21_inv @ Y22
        T22 = Y11 @ Y21_inv
        T = np.block([[T11, T12], [T21, T22]])
        return T
    elif returned_matrix == "T_inv":
        Y11 = Y[0:4, 0:4]
        Y12 = Y[0:4, 4:8]
        Y21 = Y[4:8, 0:4]
        Y22 = Y[4:8, 4:8]
        try:
            Y12_inv = np.linalg.inv(Y12)
        except:
            raise RuntimeError("Y_12 is not invertible. In general, the transmission matrix does not exist (e.g. for Delta-Wye transformers).")
        T11 = - Y12_inv @ Y11
        T12 = Y12_inv
        T21 = Y21 - Y22 @ Y12_inv @ Y11
        T22 = Y22 @ Y12_inv
        T_inv = np.block([[T11, T12], [T21, T22]])
        return T_inv
    else:
        raise ValueError(f"Invalid argument for returned_matrix: {returned_matrix}")


# Updated
def transformer_Y_matrix(element, returned_matrix='Y', flip_tbus_current_direction=True):
    V_primary, V_secondary, VA_rating, percent_z, connection, conn_type, y_shunt = \
        get_transformer_parameters(element)
    return _transformer_Y_matrix(
        V_primary, V_secondary, VA_rating, percent_z, connection, conn_type, 
        y_shunt=y_shunt, returned_matrix=returned_matrix, flip_tbus_current_direction=flip_tbus_current_direction
    )


def _transformer_Y_matrix_PSA(
    V_primary, V_secondary, VA_rating, percent_z, connection, conn_type,
    y_shunt=1e-7, returned_matrix='Y', flip_tbus_current_direction=True
):
    """
    Computes the 6x6 transformer admittance matrix for a 3-phase transformer.
    Implementation based on Steven Low's Power Systems Analysis textbook eqution 8.24.
    Supported transformer types: DY, YY
    """
    def sort_connection(connection):
        """
        Re-arrnage connection into the following form:
            [["**", "AN"], ["**", "BN"], ["**", "CN"]]
        """
        connection_sorted = [[], [], []]
        for c in connection:
            if c[1] == "AN":
                connection_sorted[0] = [c[0], "AN"]
            elif c[1] == "NA":
                connection_sorted[0] = [c[0][::-1], "AN"]
            elif c[1] == "BN":
                connection_sorted[1] = [c[0], "BN"]
            elif c[1] == "NB":
                connection_sorted[1] = [c[0][::-1], "BN"]
            elif c[1] == "CN":
                connection_sorted[2] = [c[0], "CN"]
            elif c[1] == "NC":
                connection_sorted[2] = [c[0][::-1], "CN"]
        return connection_sorted
    if conn_type[:4] == "3PDY":
        E_wc_pri = V_primary
        E_wc_sec = V_secondary / np.sqrt(3)
    elif conn_type[:4] == "3PYY":
        E_wc_pri = V_primary
        E_wc_sec = V_secondary
    else:
        raise RuntimeError(f"Not yet implemented transformer type: {conn_type}")

    y_series = (VA_rating / 3) / (percent_z * E_wc_pri ** 2)
    a_turns_ratio = E_wc_pri / E_wc_sec
    
    Y_YY = np.block([
        [y_series * np.eye(3), -a_turns_ratio * y_series * np.eye(3)], 
        [-a_turns_ratio * y_series * np.eye(3), a_turns_ratio ** 2 * (y_series + y_shunt)* np.eye(3)]
    ])
    if conn_type[:4] == "3PYY":
        if flip_tbus_current_direction:
            Y_YY = np.diag([1, 1, 1, -1, -1, -1]) @ Y_YY
        return Y_YY
    elif conn_type[:4] == "3PDY":
        port_idx = "ABCN"
        connection = sort_connection(connection)
        Gamma_I = np.block([[np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]])
        for i, conn in enumerate(connection):
            Gamma_I[i, port_idx.index(conn[0][0])] = 1
            Gamma_I[i, port_idx.index(conn[0][1])] = -1
        Gamma_T_I = np.block([[np.zeros((3, 3)), np.zeros((3, 3))],[np.zeros((3, 3)), np.eye(3)]])
        for i, port in enumerate("ABC"):
            for j in range(3):
                if connection[j][0][0] == port:
                    Gamma_T_I[i, j] = 1
                elif connection[j][0][1] == port:
                    Gamma_T_I[i, j] = -1
        Y_DY = Gamma_T_I @ Y_YY @ Gamma_I
        if flip_tbus_current_direction:
            Y_DY = np.diag([1, 1, 1, -1, -1, -1]) @ Y_DY
        return Y_DY

    
def get_line_parameters(element, cable_info=utils.CSVMapper(FILE_PATHS["cable_info"])):
    """
    Given network file line element, return standardized parameters for use of line_Y_matrix.
    Situations not handled:
        - different materials for phase vs. ground conductors
        - multiple strands for one conductor
        - we assume 3-phase, 4-wire
        - cables are all single cables for each phase/ground
        - we assume the cables are bundled in a triangular fashion
        - calculate length from waypoint when length is not given directly
        - when calculating conductor diameter, the
        - any cable type/missing info in FILE_PATHS['cable_info'] file
        - conductor temperature is assumed to be 25C

    :param element: dict, network data Line element, see net2graph
    :return: tuple, see comments below
    """

    def parse_cable_size(s):
        CONVERSION = {"4/0": "0000", "3/0": "000", "2/0": "00", "1/0": "0"}
        try:
            size, unit = re.findall(r"[\d.,/]+", s)[0], re.findall(r"[A-a]+$", s)[0]
            if size in CONVERSION:
                size = CONVERSION[size]
            assert unit.lower() in ("mcm", "kcmil", "awg")
        except:
            raise RuntimeError(
                f"Malformed or missing entry for element --> conductor --> phase --> size. Element: {element}"
            )
        return size, unit

    DEFAULT_MATERIAL = "Cu"
    has_ground = "ground" in element["line_rating"]

    # Conductor material
    material = utils.dict_get(
        element, ["line_rating", "phase", "material"], raise_err=False
    )
    material = material or DEFAULT_MATERIAL

    # Get conductor info for GMR and resistnace
    s = utils.dict_get(element, ["line_rating", "phase", "size"], raise_err=True)
    phase_size, phase_unit = parse_cable_size(s)
    if has_ground:
        s = utils.dict_get(element, ["line_rating", "ground", "size"], raise_err=True)
        gnd_size, gnd_unit = parse_cable_size(s)

    # GMR (unit: ft)
    phase_gmr = cable_info.map(phase_unit, "GMR (ft)", phase_size, print_err=True)
    assert (
        len(phase_gmr) > 0
    ), f"Missing or duplicate GMR data for {phase_size} {phase_unit}: {phase_gmr}"
    phase_gmr = float(phase_gmr.pop())
    if has_ground:
        gnd_gmr = cable_info.map(gnd_unit, "GMR (ft)", gnd_size, print_err=True)
        assert (
            len(gnd_gmr) > 0
        ), f"Missing or duplicate GMR data for {gnd_size} {gnd_unit}: {gnd_gmr}"
        gnd_gmr = float(gnd_gmr.pop())
        gmr = np.array([phase_gmr, phase_gmr, phase_gmr, gnd_gmr])
    else:
        gmr = np.array([phase_gmr, phase_gmr, phase_gmr])

    # # radii (unit: ft)
    # phase_conductor_radius = cable_info.map(phase_unit, 'Diameter (in)', phase_size, print_err=True)
    # gnd_conductor_radius = cable_info.map(gnd_unit, 'Diameter (in)', gnd_size, print_err=True)
    # radii = np.array([phase_conductor_radius, phase_conductor_radius, phase_conductor_radius, gnd_conductor_radius])

    # Resistance per length (unit: Ohm/mile) at 25 degrees C
    column_name = f"{material} Resistance AC 60Hz 20C (Ohm/1000 ft)"
    phase_strands = str(
        utils.dict_get(element, ["line_rating", "phase", "strands"], raise_err=True)
    )
    phase_resistance = cable_info.map(
        (phase_unit, "Strands"),
        column_name,
        (phase_size, phase_strands),
        print_err=True,
    )
    assert (
        len(phase_resistance) > 0
    ), f"Missing or indeterminate resistance data for {phase_size} {phase_unit}: {phase_resistance}"
    phase_resistance = float(phase_resistance.pop())
    if has_ground:
        gnd_strands = str(
            utils.dict_get(
                element, ["line_rating", "ground", "strands"], raise_err=True
            )
        )
        gnd_resistance = cable_info.map(
            (gnd_unit, "Strands"), column_name, (gnd_size, gnd_strands), print_err=True
        )
        assert (
            len(gnd_resistance) > 0
        ), f"Missing or indeterminate resistance data for {gnd_size} {gnd_unit}: {gnd_resistance}"
        gnd_resistance = float(gnd_resistance.pop())
        resistance = [
            phase_resistance,
            phase_resistance,
            phase_resistance,
            gnd_resistance,
        ]
    else:
        resistance = [phase_resistance, phase_resistance, phase_resistance]
    # Unit conversion: Ohm/1000ft --> Ohm/mile
    resistance = np.array(resistance) * 5.280

    # Cable cross-sectional arrangement (unit: ft)
    cable_diameter = utils.dict_get(
        element, ["line_rating", "phase", "o.d."], raise_err=True
    )
    dist_from_orig = cable_diameter / np.sqrt(3) / 12
    if has_ground:
        cable_locs = dist_from_orig * np.array(
            [
                np.exp(0 * 1j * np.pi),
                np.exp(-2 / 3 * 1j * np.pi),
                np.exp(2 / 3 * 1j * np.pi),
                0,
            ]
        )
    else:
        cable_locs = dist_from_orig * np.array(
            [
                np.exp(0 * 1j * np.pi),
                np.exp(-2 / 3 * 1j * np.pi),
                np.exp(2 / 3 * 1j * np.pi),
            ]
        )

    # Overall cable length (unit: mile)
    length = utils.dict_get(element, ["length"], raise_err=True) / 5280

    return length, cable_locs, gmr, resistance


# Updated
def _line_Y_matrix(length, cable_locs, gmr, resistance, f=F, zero_shunt=False, returned_matrix='Y', flip_tbus_current_direction=True):
    """
    Given line parameters, generate its admittance matrix or transmission matrix.
    Reference:
        Kersting, W. H. (2012). Distribution system modeling and analysis, Third
        Edition (3rd ed., revised). CRC Press, Taylor & Francis Group [distributor].
    Calculation for phase impedance matrix is given in Example 4.1 (p.92) with
    variable defined on p.81
    Calculation for shunt admittance matrix is in Chaper 5 (eqn 5.9, 5.10, 5.15),
    but they are not implemented.
    The overall 8x8 system line model is given in Chapter 5 (eqn 6.19)
    Note: The overall structure is correct, but the shunt admittance and series
        inductance terms depend on accuracy cable configuration (e.g. sizing, tape shield, concentric). 
        The dominant factor is cable resistance in common distribution circuits.
    Warning: Due to small and potentially inaccuracy shunt term, T matrix is 
        preferred over Y matrix when specifying constraints. 
    Note: Neutral is retained. If neutral should be removed, perform Kron reduction.

    :param length: float, unit: mile, overall cable length
    :param cable_locs: np.ndarray, np.csingle, unit: ft, shape: (# of cables).
        Cable location in cartesian coordiante, with the real component being
        the x axis and the imaginary component the y axis.
    :param gmr: np.ndarray, float, unit: ft, shape: (# of cables), geometric
        mean radii of cable. This is generally given in the specs sheet of
        the cable and depends on the cable cross-sectional area and stranding.
    :param resistance: np.ndarray, float, unit: Ohm/mile, shape: (# of cables).
        Resistance values of each conductor.
    :param zero_shunt: bool, whether to ignore shunt admittance term
    :param returned_matrix: str, one of {'Y', 'T', 'T_inv'}, whether to return the
        - admittance matrix Y: [I1, I2] = Y @ [V1, V2]
        - transmission matrix T: [I1, V1] = T @ [I2, V2]
        - transmission matrix T inverse: [I2, V2] = T_inv @ [I1, V1]
    :param f: float, frequency
    :return: np.ndarray, float, 8x8 matrix. The I, V vectors are A, B, C, N.
    """
    rho_earth = 100  # Earth's resistivity (Ohm-m)
    assert len(gmr) == len(resistance) == len(cable_locs), (
        len(gmr),
        len(resistance),
        len(cable_locs),
    )
    W = len(gmr)
    assert W in (3, 4), W

    # Construct series impedance matrix Z (Kersting Equation 4.39, 4.40)
    D = np.zeros((W, W), dtype=np.csingle)
    for i in range(W):
        for j in range(W):
            D[i, j] = abs(cable_locs[i] - cable_locs[j])
    D = D + np.diag(gmr)
    Z = (
        0.00158836 * f
        + 1j * 0.00202237 * f * (-np.log(D) + 7.6786 + 0.5 * np.log(rho_earth / f))
        + np.diag(resistance)
    )
    Z *= length  # shape = (W, W)
    if returned_matrix == 'series Z':
        return Z
    # Construct shunt admittance matrix Y (Kersting Equation 5.9, 5.10, 5.15, 5.31, 5.32, 5.38)
    # Shunt to ground
    if zero_shunt:
        Y_shunt = np.zeros((W,W), dtype=np.csingle)
    else:
        # Adapted from Eqn 5.38, a guess. We assume tape shield with 133% insulation thickness.
        # Requires cable type (tape shield/concentric neutral/overhead), 
        # and if underground: insulation thickness, conductor radius, 
        Y_shunt = np.eye(W) * 77.3619j / np.log(1.3) * 1e-6 * length

    a = np.eye(W) + 0.5 * Z @ Y_shunt
    b = Z
    c = Y_shunt + 0.25 * Y_shunt @ Z @ Y_shunt
    d = a
    if returned_matrix == 'T':
        # Equation 6.19
        T = np.block([[a, b], [c, d]])
        if not flip_tbus_current_direction:
            T = T @ np.diag([-1, -1, -1, -1, 1, 1, 1, 1])
        return T
    elif returned_matrix == 'T_inv':
        # Equation 6.20
        T_inv = np.block([[d, -b], [-c, a]])
        if not flip_tbus_current_direction:
            T_inv = np.diag([-1, -1, -1, -1, 1, 1, 1, 1]) @ T_inv
        return T_inv
    elif returned_matrix == 'Y':
        # Requires c to be nonsingular
        assert not zero_shunt, "Shunt admittance must be nonzero in order to calculate Y matrix."
        c_inv = inv(c)
        Y = np.block([[-c_inv @ d, c_inv], [b - a @ c_inv @ d, a @ c_inv]])
        if not flip_tbus_current_direction:
            Y = np.diag([1, 1, 1, 1, -1, -1, -1, -1]) @ Y
        return Y
    elif returned_matrix == "component Y":
        # Equation 6.3 / Figure 6.1 in Kersting.
        if zero_shunt:
            return inv(Z)
            # return inv(np.diag(np.diag(Z)))
        else:
            raise RuntimeError("not implemented")
    else:
        raise ValueError(f"Invalid argument for returned_matrix: {returned_matrix}")


# Updated
def line_Y_matrix(element, cable_info=utils.CSVMapper(FILE_PATHS['cable_info']), zero_shunt=True, returned_matrix='Y', flip_tbus_current_direction=True, print_info=True):
    # If line rating is missing or it is not in our dictionary format, assume zero-impedance.
    if (
        ("line_rating" not in element)
        or (not element["line_rating"])
        or (type(element["line_rating"]) is str)
        or ("length" not in element)
        or (element["length"] == 0)
    ):
        if print_info: print(f"[Warning] Missing line parameter for {element['name']}. Assuming zero-impedance.")
        return np.identity(8 if element['phases'] == 'abc' else len(element['phases'])*2), True
    length, cable_locs, gmr, resistance = get_line_parameters(element, cable_info=cable_info)
    return _line_Y_matrix(length, cable_locs, gmr, resistance, f=F, zero_shunt=zero_shunt, returned_matrix=returned_matrix, flip_tbus_current_direction=flip_tbus_current_direction), False
    

# Updated
def network_Y_matrix(edges, node_list, net_data):
    """
    Given an electrical network graph, return its admittance matrix Y.
    Only 3-phase transformers and lines are implemented.
    return: np.ndarray, float, 3N x 3N matrix, where N is the number of nodes.
    """
    N = len(node_list)
    Y = np.zeros((N, N, 3, 3), dtype=np.csingle)
    for edge in edges.values():
        j, k = node_list.index(edge["source"]), node_list.index(edge["target"])
        if edge['element_type'] in ('Line', 'Transformer'):
            zero_impedance = False
            if edge['element_type'] == 'Line':
                component_Y, zero_impedance = line_Y_matrix(get_element(edge, net_data), returned_matrix='Y', zero_shunt=False, flip_tbus_current_direction=False)
                # todo: remove
                # component_Y = np.real(component_Y)
                # component_Y *= 1e2
                if zero_impedance:
                    raise RuntimeError("Electrical graph should not contain any zero impedance line.")
            else:
                transformer_element = get_element(edge, net_data)
                if transformer_element['transformer_type'] not in ('3PDYG', '3PDY', '3PYYG', '3PYY'):
                    raise RuntimeError(f"Unknown transformer type: {transformer_element['transformer_type']}")
                component_Y = transformer_Y_matrix(transformer_element, returned_matrix='Y', flip_tbus_current_direction=False)
        else:
            # All other edge types (zero or infinite impedance) are not supported.
            raise ValueError(f"Unknown element type: {edge['element_type']}")
        component_Y = math_utils.matrix2block(component_Y, 4)[:,:,:3,:3]
        edge['component_Y'] = math_utils.block2matrix(component_Y)
        # This is paper equation (6).
        Y[j,j] += component_Y[0, 0]
        Y[k,k] += component_Y[1, 1]
        Y[j,k] += component_Y[0, 1]
        Y[k,j] += component_Y[1, 0]
    Y = math_utils.block2matrix(Y)
    return Y


"""Plotting"""


# This is updated
def plot_graph(
    nodes,
    edges,
    decorators,
    out_path=None,
    G=None,
    node_positions=None,
    decorator_positions=None,
    layout=nx.planar_layout,
    figsize=None,
    edge_attr=None,
    with_labels=True,
    edge_widths=None,
    directed=True,
    plot_connected_components=False,
    txt=None,
    txt_position=None,
):
    # Make networkx graph
    node_list, edge_list, edge_info, edge_dict = (
        list(nodes.keys()),
        to_edge_list(edges),
        to_edge_dict(edges, flip=True),
        to_edge_dict(edges, flip=False),
    )
    if not G:
        G = nx.DiGraph(edge_list) if directed else nx.Graph(edge_list)
        # Add disconnected nodes
        for n in set(node_list) - set(utils.flatten_iterable(edge_list, 2)):
            G.add_node(n)
    else:
        edge_info = {edge: "" for edge in G.edges()}

    # Add decorators: makeshift plotting (add them as nodes and add connecting edge)
    edge_midpoint_temp_nodes = set()
    midpoint_nodes_positions = {}
    cnt = 0
    for name, decorator in decorators.items():
        G.add_node(name)
        for decorated_name, decorated_type in decorator["decorated_elements"].items():
            if decorated_name in G.nodes:
                G.add_edge(name, decorated_name)
                cnt += 1
            elif edge_dict[decorated_name] in G.edges:
                temp_node_name = f"edge_midpoint_temp_node_{cnt}"
                G.add_node(temp_node_name)
                G.add_edge(name, temp_node_name)
                edge_midpoint_temp_nodes.add(temp_node_name)
                if node_positions and decorator_positions:
                    midpoint_nodes_positions[temp_node_name] = decorator_positions[
                        name
                    ][2:4]
                cnt += 1

    # Format node size (make edge_midpoint_temp_nodes small)
    size_mul = (figsize[0] if figsize else 12) / 12
    node_sizes = np.ones(len(G.nodes)) * 100  * size_mul
    if decorators:
        for n, i in zip(G.nodes, range(len(G.nodes))):
            if n in edge_midpoint_temp_nodes:
                node_sizes[i] = 10
    # Format node positions (either computed by us or by networkx)
    if node_positions:
        if decorator_positions:
            node_positions.update(
                {name: pos[0:2] for name, pos in decorator_positions.items()}
            )
            node_positions.update(midpoint_nodes_positions)
        node_positions = {k: np.array(node_positions[k]) for k in node_positions}
    else:
        try:
            if layout == 'tree':
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="")
                node_positions = {k: (v[0], v[1] + (random.random() - 0.5) * 40) for k, v in pos.items()}
            else:
                node_positions = layout(G)
        except:
            print("Node position layout error. Falling back to spring layout.")
            node_positions = nx.spring_layout(G)
    # Format edge attributes
    if edge_attr:
        for edge, attr in edge_attr.items():
            edge_info[edge] = str(edge_info[edge]) + str(attr)
    if edge_widths:
        edge_widths = [edge_widths[edge] if edge in edge_widths else edge_widths[(edge[1], edge[0])] for edge in G.edges()]
    # Plot
    fig = plt.figure(figsize=figsize)
    nx.draw_networkx_edge_labels(
        G, node_positions, edge_labels=edge_info, font_color="red", font_size=18*size_mul
    )
    nx.draw_networkx(
        G, node_positions, arrows=True, node_size=node_sizes, width=edge_widths, 
        with_labels=with_labels, font_size=12*size_mul
    )
    if txt:
        plt.text(*txt_position, txt, fontsize=12, transform=fig.transFigure)
    plt.axis("off")
    plt.tight_layout()
    if out_path:
        utils.mkdir(os.path.dirname(out_path))
        plt.savefig(out_path, bbox_inches="tight")
    if plot_connected_components:
        cc = list(nx.weakly_connected_components(G))
        for i, c in enumerate(cc):
            H = G.subgraph(c)
            plot_nx_graph(H, os.path.join(os.path.dirname(out_path), f"connected_component_{i}.png"), edge_info=edge_info, figsize=figsize)


# This is updated
def plot_nx_graph(G, outpath=None, show=True, edge_info=None, figsize=(12,12)):
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="")
    except: 
        print("graphviz not installed or error. Using spring layout.")
        pos = nx.spring_layout(G)
    pos = {k: (v[0], v[1] + (random.random() - 0.5) * 40) for k, v in pos.items()}
    fig = plt.figure(figsize=figsize)
    if edge_info:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: edge_info[k] for k in G.edges}, font_color="red")
    nx.draw(G, pos, with_labels=True)
    fig.tight_layout()
    if outpath:
        plt.savefig(outpath)
    if show:
        plt.show()
    else:
        plt.close()


def Y_matrix2graph(Y, W=3, svd_thresh=1e-10, return_type='edge_list'):
    """
    An edge is present if the matrix is non-zero.
    :param Y: np.ndarray, shape=(3n, 3n), admittance matrix.
    :param W: int, block size.
    :param svd_thresh: float, threshold for considering an edge to exist.
    :param return_type: str, one of 'edge_list', 'adjacency'.
    :returns: depends on return_type. We use zero-based indexing.
    """
    N = Y.shape[0] // W
    incidence_matrix = np.zeros((N, N))
    edge_list = []
    Y_blk = math_utils.matrix2block(Y, W)
    for i in range(N):
        for j in range(i+1, N):
            # if (np.linalg.svd(Y_blk[i,j])[1]).min() > thresh:     # Matrix must be full-rank
            if (np.linalg.svd(Y_blk[i,j])[1]).max() > svd_thresh:       # Matrix must be non-zero
                if return_type == 'edge_list':
                    edge_list.append((i, j))
                else:
                    incidence_matrix[i,j] = 1
                    incidence_matrix[j,i] = 1
    if return_type == 'edge_list':
        return edge_list
    else:
        return incidence_matrix


def plot_Y_matrix(
    Y, 
    filename=None, 
    W=3, 
    plot_mtx=True, 
    svd_thresh=1e-10, 
    svd_as_width=True, 
    node_labels=None,
    figsize=(12, 12),
    layout='tree'
):
    """
    Generate the graph corresponding to the admittance matrix Y.
    Plot the resulting graph.
    :param W: int, block size, number of wires (default 3-phase 3-wire).
    """
    edge_list = Y_matrix2graph(Y, W=W, return_type='edge_list', svd_thresh=svd_thresh)
    nodes = {n: n for n in set(np.array(edge_list).reshape(-1))}
    edges = {e: {'source': i, 'target': j} for e, (i, j) in enumerate(edge_list)}
    Y_blk = math_utils.matrix2block(Y, W)
    edge_attr = {(i, j): Y_blk[i,j] for i,j in edge_list} if plot_mtx else None       
    edge_widths = None
    if svd_as_width:
        edge_widths = {(i, j): np.linalg.svd(Y_blk[i,j])[1].mean() for i, j in edge_list}
        svd_max = np.array(list(edge_widths.values())).max()
        edge_widths = {k: max(v / svd_max * 8, 1) for (k, v) in edge_widths.items()}
    txt, txt_position = None, None
    if node_labels:
        txt = '\n'.join([f"{i}: {n}" for i, n in enumerate(node_labels)])
        txt_position = (0.05, 0.1)
    plot_graph(
        nodes, edges, {}, out_path=filename, directed=False, edge_attr=edge_attr, 
        figsize=figsize, edge_widths=edge_widths, txt=txt, txt_position=txt_position, layout=layout
    )
