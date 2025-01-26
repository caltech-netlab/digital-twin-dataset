import os
import sys
import pathlib
import copy
from datetime import datetime
from dateutil import parser
import networkx as nx
import numpy as np
import cvxpy as cp
from tqdm import tqdm

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]) + '/utils')
import utils
import math_utils
import topology
import phasor_utils
from global_param import FILE_PATHS


class StateEstimator:
    def __init__(
        self,
        network_files, 
        network_metadata_dir=FILE_PATHS['net_metadata'], 
        input_data_dir=FILE_PATHS['phasors'], 
        output_data_dir='temp/state_estimation', 
        phase_ref=None, 
        delta_t_threshold=5., 
    ):
        """
        :param network_files: list of str, paths to network files. State estimation will be done for
            ALL elements in the network files.
        :param datetimespan: list of 2 str, see utils.read_ts
        :param input_data_dir: str, path to egauge waveform data
        :param phase_ref: str, name of network element (with phase a/b/c) as 0-degree reference.
        :param delta_t_threshold: float, see phasor_utils.align_phasors
        :param print_info: bool
        """
        self.network_files = network_files
        self.network_metadata_dir = network_metadata_dir
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.phase_ref = phase_ref
        self.delta_t_threshold = delta_t_threshold
        self.element_parents = utils.dict_parents(utils.load_json(FILE_PATHS["element_inheritance"], raise_err=True))
        self.load_topology_changes()

    def load_topology_changes(self, network_files=None):
        network_files = network_files or self.network_files
        # Load topology metadata
        if all([f.endswith('.json') for f in network_files]):
            self.topology_changes = [None]
        else:
            self.topology_changes = topology.get_metadata(metadata_path=self.network_metadata_dir)[2]    
    
    def load_topology_data(self, t0, network_files, input_data_dir, print_info):
        """
        Load ALL measured quantities in network_files.
        """
        # Load topology data
        if type(t0) is str:
            t0 = parser.parse(t0)
        net_data = topology.load_net_files(network_files, t=t0)
        nodes, edges, decorators = topology.net2graph(
            net_data, self.element_parents, fix_implicit=False, connected_edges=True, print_warning=True
        )
        assert not topology.check_duplicate_edge(edges), "Duplicate edge is not currently supported."

        # Load meter information
        # input_files: element.phase --> path to file; unit_lookup: element.phase --> unit
        input_files, unit_lookup, nominal_values = {}, {}, {}
        for name, decorator in decorators.items():
            if decorator['element_type'] != 'EgaugeMeter': continue
            meter = topology.get_element(decorator, net_data)
            for reg in meter['registers']:
                if (reg['unit'] not in ('V', 'A')) or ('data_file' not in reg) or ('element' not in reg):
                    continue
                if reg['unit'] == 'A':
                    if 'I_rating' in reg:
                        nominal_values[reg['element'].split('.')[0]] = reg['I_rating']
                    else:
                        print(f"[WARNING] Missing CT I_rating for register {reg['name']}. Data discarded.")
                        continue
                else:
                    if 'nominal_voltage' in net_data['Bus'][reg['element'].split('.')[0]]:
                        nominal_values[reg['element'].split('.')[0]] = net_data['Bus'][reg['element'].split('.')[0]]['nominal_voltage']
                    else:
                        print(f"[WARNING] Missing nominal_voltage for element {reg['element']}. Data discarded.")
                        continue
                input_files[reg['element']] = os.path.join(input_data_dir, f"{name}-{reg['value']}")
                unit_lookup[reg['element']] = reg['unit']
        return net_data, nodes, edges, decorators, input_files, unit_lookup, nominal_values
    
    def _state_estimation_BIM(
        self, net_data, nodes, edges, decorators, time_col, topology_timestamp,
        metered_df, unit_lookup, nominal_values, output_data_dir, print_info, write_mode, 
        keep_cc=None, prune_nodes=None, prune_edges=None, measured_injections={}, plot=False,
    ):
        """
        This is the core state estimation algorithm based on the bus injection model (BIM).
        Since we do not measure nodal current injection directly, 
        we need to provide this (hardcoded) via measured_injections.
        To do so, first run the function with plot=True and examine the network graph.
        Then, provide the measured injections in the form of a dictionary.

        :prune_nodes: list of nodes to prune from the electrical network
        :prune_edges: list of edges to prune from the electrical network
        :measured_injections: dictionary of measured injections, e.g.
            {
                "bus_1234|bus_1235": [
                    {"element": "tr_1", "sign":+1}, 
                    {"element": "line_2", "sign":-1}, 
                ],
                ...
            }
        :plot: bool, whether to plot the network graph
        """
        # Build electrical graph and prune extraneous elements
        if type(output_data_dir) is not list:
            output_data_dir = [output_data_dir]
        nodes_e, edges_e, decorators_e = topology.graph2electrical_graph(
            copy.deepcopy(nodes),
            copy.deepcopy(edges),
            {},
            net_data,
            self.element_parents,
        )
        if plot:
            topology.plot_graph(
                nodes_e, edges_e, decorators_e, "temp/state_estimation_plots/original_network/electrical_topology.png",
                layout=nx.spring_layout, figsize=(12, 8), plot_connected_components=True
            )
        if keep_cc:
            edge_info = topology.to_edge_dict(edges_e, flip=False)
            G = nx.DiGraph(topology.to_edge_list(edges_e))
            cc = nx.weakly_connected_components(G)
            keep_nodes = utils.flatten_iterable([list(c) for i, c in enumerate(cc) if i in keep_cc], 2)
            G = G.subgraph(keep_nodes)
            nodes_e = {k: v for k, v in nodes_e.items() if k in G.nodes}
            edges_e = {k: v for k, v in edges_e.items() if edge_info[k] in G.edges}
            if print_info: print(G.nodes)
        if prune_nodes:
            for node in prune_nodes:
                del nodes_e[node]
                to_delete = [e for e, el in edges_e.items() if node in (el['source'], el['target'])]
                for name in to_delete:
                    del edges_e[name]
        if prune_edges:
            for edge in prune_edges:
                del edges_e[edge]
        node_list = list(nodes_e.keys())
        N, E = len(nodes_e), len(edges_e)
        if print_info: 
            print(f"Number of nodes: {N}\nNumber of edges: {E}")
            print("Nodes:\n ", '\n  '.join(node_list))
        if plot:
            topology.plot_graph(
                nodes_e, edges_e, decorators_e, "temp/state_estimation_plots/pruned_network/electrical_topology.png",
                layout='tree', figsize=(12, 8), plot_connected_components=True
            )

        # Load V, I measurements
        V_metered, I_metered, I_metered_nominal, V_metered_nominal = [], [], [], []
        V_metered_nodes, I_metered_nodes, V_metered_idx = [], [], []
        constraints = []
        # Pre-process metered_df (gather phase information)
        metered_df_info = {}
        for element_phase in metered_df:
            element, phase = element_phase.split('.')
            utils.insert_dict_list(metered_df_info, [element], phase)
        # Re-order nodes so nodes with measured currents come first
        node_list = sorted(node_list, key=lambda x: 0 if x in measured_injections else 1)
        for idx, node in enumerate(node_list):
            # V meaurement: detect automatically
            for name in node.split('|'):
                if name in metered_df_info:
                    phases = metered_df_info[name]
                    if set(phases) - {'ag', 'bg', 'cg', 'ng'}:
                        print(f"[Warning] Unknown phase configuration: {phases} for {name}")
                        continue
                    measurement = [
                        phasor_utils.polar2rectangular(
                            metered_df[f"{name}.{p}"]['phase'], 
                            metered_df[f"{name}.{p}"]['rms']
                        )
                        for p in ('ag', 'bg', 'cg')
                    ]
                    V_metered.append(np.stack(measurement, axis=0))
                    V_metered_nominal += [nominal_values[name] for _ in sorted(phases)][:3]
                    V_metered_nodes.append(node)
                    V_metered_idx += [idx * 3 + p for p in range(3)]
                    break
            # I measurement: supplied by user (i.e. hardcoded) via measured_injections
            if node in measured_injections:
                currents, nominal = [], []
                if len(measured_injections[node]) == 0:
                    # Option 1: Soft constraint
                    I_metered.append(np.zeros((3, len(time_col)), dtype=np.csingle))
                    I_metered_nominal.append(np.ones(3))
                    I_metered_nodes.append(node)
                    # Option 2: Hard constraint
                    # constraints.append()
                    continue
                for d in measured_injections.pop(node):
                    phases = metered_df_info[d['element']]
                    if set(phases) - {'a', 'b', 'c', 'n'}:
                        print(f"[Warning] Unknown phase configuration: {phases} for {name}")
                        continue
                    measurement = [
                        phasor_utils.polar2rectangular(
                            metered_df[f"{d['element']}.{p}"]['phase'], 
                            metered_df[f"{d['element']}.{p}"]['rms']
                        )
                        for p in 'abc'
                    ]
                    currents.append(np.stack(measurement, axis=0) * d['sign'])
                    nominal.append([nominal_values[d['element']] for _ in 'abc'])
                I_metered.append(np.stack(currents, axis=0).sum(axis=0))
                I_metered_nominal.append(np.mean(np.array(nominal), axis=0))
                I_metered_nodes.append(node)
        MI, MV, T = len(I_metered), len(V_metered), len(time_col)
        I_metered = np.stack(I_metered, axis=0).reshape(MI * 3, T)
        V_metered = np.stack(V_metered, axis=0).reshape(MV * 3, T)
        I_metered_nominal = np.array(I_metered_nominal).flatten()
        V_metered_nominal = np.array(V_metered_nominal).flatten()

        # Build network admittance matrix
        Y = topology.network_Y_matrix(edges_e, node_list, net_data)

        # Solve state estimation: iterate over time
        if print_info: print("Solving...")
        T = len(time_col)
        V_hat = np.empty((N*3, T), dtype=np.csingle)
        I_hat = np.empty((N*3, T), dtype=np.csingle)
        for t in tqdm(range(T)):
            V_hat_t = cp.Variable(N*3, complex=True)
            # Objective function: minimize l2 distance between metered and computed I/V values
            # Here, we normalize by CT ratings and nominal bus voltages.
            I_residual = cp.abs(I_metered[:, t] - Y[:MI*3] @ V_hat_t)
            V_residual = cp.abs(V_metered[:, t] - V_hat_t[V_metered_idx])
            obj_fxn = cp.sum(I_residual / I_metered_nominal) + cp.sum(V_residual / V_metered_nominal)
            obj = cp.Minimize(obj_fxn)
            prob = cp.Problem(obj, constraints)
            try:
                prob.solve(solver=cp.ECOS)
            except Exception as e:
                print(f"Error at {time_col[t]}: {e}")
                continue
            V_hat[:,t] = np.array(V_hat_t.value).flatten()
            I_hat[:,t] = np.array((Y @ V_hat_t).value).flatten()

        # Write nodal results to disk
        soln_lookup = {}
        for n, name in enumerate(node_list):
            df_I = {'t': time_col}
            df_V = {'t': time_col}
            for p, phase in enumerate('abc'):
                df_I[phase] = I_hat[3*n + p]
                df_V[phase] = V_hat[3*n + p]
            for d in output_data_dir:
                utils.write_csv_ts(df_I, os.path.join(d, name[:128] + "-I"), mode=write_mode)
                utils.write_csv_ts(df_V, os.path.join(d, name[:128]), mode=write_mode)
            if name in I_metered_nodes:
                j = I_metered_nodes.index(name)
                measurement_df = {p: I_metered[j*3 + i] for i, p in enumerate('abc')} | {'t': time_col}
                utils.write_csv_ts(measurement_df, os.path.join(d, name[:128] + "-I-measurement"), mode=write_mode)
            if name in V_metered_nodes:
                j = V_metered_nodes.index(name)
                measurement_df = {p: V_metered[j*3 + i] for i, p in enumerate('abc')} | {'t': time_col}
                utils.write_csv_ts(measurement_df, os.path.join(d, name[:128] + "-measurement"), mode=write_mode)
            # Write voltage metadata (solutions lookup)
            payload = {
                'file': name[:128],
                'phases': 'abc',
                'sign': 1,
                "metered": name in V_metered_nodes,
                'zero': {"a": False, "b": False, "c": False},
            }
            for el_name in name.split('|'):
                soln_lookup[el_name] = payload
            # Write current metadata (solutions lookup)
            soln_lookup[name + "-I"] = {
                'file': name[:128] + "-I",
                'phases': 'abc',
                'sign': 1,
                "metered": name in I_metered_nodes,
                'zero': {"a": False, "b": False, "c": False},
            }

        # Solve branch current by 
        # 1) iterating over all non-zero-impedance edges
        for name, edge in edges_e.items():
            j, k = node_list.index(edge["source"]), node_list.index(edge["target"])
            V_j, V_k = V_hat[j*3:(j+1)*3], V_hat[k*3:(k+1)*3]
            I = edge['component_Y'] @ np.concatenate([V_j, V_k], axis=0)
            df_fbus = {p: I[i] for i, p in enumerate('abc')} | {'t': time_col}
            df_tbus = {p: I[3+i] for i, p in enumerate('abc')} | {'t': time_col}
            utils.write_csv_ts(df_fbus, os.path.join(d, f"{name[:128]}-fbus"), mode=write_mode)
            utils.write_csv_ts(df_tbus, os.path.join(d, f"{name[:128]}-tbus"), mode=write_mode)
            soln_lookup[name] = {
                "fbus": {
                    "file": f"{name[:128]}-fbus",
                    "phases": "abc",
                    "sign": 1,
                    "metered": None,       # We don't specify
                    "zero": {"a": False, "b": False, "c": False}
                },
                "tbus": {
                    "file": f"{name[:128]}-tbus",
                    "phases": "abc",
                    "sign": 1,
                    "metered": None,       # We don't specify
                    "zero": {"a": False, "b": False, "c": False}
                }
            }
        # Optionally, compute KCL at all nodes to recover the branch currents in the original graph
        
        # Save metadata
        for d in output_data_dir:
            utils.save_json(os.path.join(d, 'metadata', f"{topology_timestamp}.json"), soln_lookup)
        last_timestamp = time_col[-1]
        if last_timestamp:
            for d in output_data_dir:
                utils.save_json(os.path.join(d, 'metadata', 't.json'), {'last_timestamp': str(last_timestamp)})
        
    def _state_estimation_BFM(
        self, net_data, nodes, edges, decorators, time_col, topology_timestamp,
        metered_df, unit_lookup, nominal_values, output_data_dir, print_info, write_mode
    ):
        """
        This is the core state estimation algorithm based on the branch flow model (BFM).
        """
        # Private helper functions below
        def merge_variables(names, var_names, name_src, name_tgt, equal_neg=None):
            """
            For network elements that have the same voltage/current value either from:
            - KCL in a chain --> currents equal
            - zero-imepdance edge --> fbus/tbus currents equal and src/tgt voltages equal
            we reduce the number of redundant decision variables.
            """
            var_names_src = var_names[names[name_src]['idx']]
            var_names_tgt = var_names[names[name_tgt]['idx']]
            # Merge element names and phases (tgt -> src).
            # The order of var_names['elements'] is important. The first element determines the positive current direction.
            var_names_src['elements'] += var_names_tgt['elements']
            var_names_src['phases'] += ''.join([e for e in var_names_tgt['phases'] if e not in var_names_src['phases']])
            # Reset indices and sign
            if 'sign' in names[name_tgt]:
                orig_sign = names[name_tgt]['sign']
                new_sign = names[name_src]['sign'] * (-1 if equal_neg else 1)
            for element in var_names_tgt['elements']:
                names[element]['idx'] = names[name_src]['idx']
                if 'sign' in names[element]:
                    names[element]['sign'] = new_sign * (-1 if orig_sign != names[element]['sign'] else 1)
            # If either of src or tgt has 'zero' == True, then the merged variable has 'zero' == True.
            for k in (set(var_names_src['zero'].keys()) & set(var_names_tgt['zero'].keys())):
                var_names_src['zero'][k] = var_names_src['zero'][k] or var_names_tgt['zero'][k]
                var_names_tgt['zero'][k] = var_names_src['zero'][k] or var_names_tgt['zero'][k]
            var_names_tgt.clear()
        def name2idx(names, var_names, element_name, sign=False):
            info = var_names[names[element_name]['idx']]
            # Check if element is constrained to zero
            if 'idx' not in info:
                zero_idx = [None] * len(names[element_name]['phases'])
                return (zero_idx, 1) if sign else zero_idx
            idx = [info['idx'] + info['phases'].index(p) for p in names[element_name]['phases']]
            return idx, (names[element_name]['sign'] if sign else +1)
        def make_arr(var_arr, idx, sign):
            return cp.hstack([0 if i is None else var_arr[i] for i in idx]) * sign
        def name2arr(var_arr, names, var_names, element_name, sign=False):
            idx, sign = name2idx(names, var_names, element_name, sign)
            return make_arr(var_arr, idx, sign)

        if type(output_data_dir) is not list:
            output_data_dir = [output_data_dir]
        # Scan all current and voltage elements and list them as variables
        # I_names (corresponds to all network edge/injection elements) = {
        #   "<element>-<tbus/fbus>": {
        #     'phases': 'abcn',
        #     'idx': int, index into I_names,
        #     'sign': +1 or -1
        #   },
        #   ...
        # }
        # V_names Same as I_names, without 'sign'.
        # I_var_names (corresponds to decision variables) = [
        #   {
        #       'elements': [element name, element name, ...],
        #       'phases': 'abcn',
        #       'zero': {'a': False, 'b': False, 'c': False, 'n': True}, 
        #           (whether the element is constrained to be zero)
        #       'idx': int (index into I, set when declaring I array),
        #   },
        #   ...
        # ]
        # V_var_names is the same as I_var_names.
        # The key 'phases' appears in both places because the I_var_names may have more phases 
        #   than the I_names. e.g. a 3-phase transformer serving a single-phase circuit.
        if print_info: print("Building optimization problem...")
        I_var_names, V_var_names, I_names, V_names, I_idx, V_idx, I_idx_p, V_idx_p = [], [], {}, {}, 0, 0, 0, 0
        for name, edge in edges.items():
            # Note: Consider adding support 3-phase 3-wire to slightly increase efficiency.
            #   Currently, our T matrix is 8x8, but it can be reduced to 6x6 for 3-phase 3-wire.
            el = topology.get_element(edge, net_data)
            phases = 'abcn' if el['phases'] == 'abc' else el['phases']
            assert phases in ['abc', 'abcn'], f"Only 3-phase is currently supported, but we have phases {phases} for {name}"
            I_names[f"{name}-fbus"] = {'phases': phases, 'idx': I_idx, 'sign': +1}
            I_var_names.append({'elements': [f"{name}-fbus"], 'phases': phases, 'zero': {p: False for p in phases}})
            I_names[f"{name}-tbus"] = {'phases': phases, 'idx': I_idx+1, 'sign': +1}
            I_var_names.append({'elements': [f"{name}-tbus"], 'phases': phases, 'zero': {p: False for p in phases}})
            I_idx += 2
            I_idx_p += len(phases) * 2
        for name, inj in decorators.items():
            if 'ACInjection' not in self.element_parents[inj['element_type']]: continue
            el = topology.get_element(inj, net_data)
            phases = 'abcn' if el['phases'] == 'abc' else el['phases']
            assert phases in ['abc', 'abcn'], f"Only 3-phase is currently supported, but we have phases {phases} for {name}"
            I_names[name] = {'phases': phases, 'idx': I_idx, 'sign': +1}
            I_var_names.append({'elements': [name], 'phases': phases, 'zero': {p: False for p in phases}})
            I_idx += 1
            I_idx_p += len(phases)
        for name, node in nodes.items():
            el = topology.get_element(node, net_data)
            phases = 'abcn' if el['phases'] == 'abc' else el['phases']
            assert phases in ['abc', 'abcn'], f"Only 3-phase is currently supported, but we have phases {phases} for {name}"
            V_names[name] = {'phases': phases, 'idx': V_idx}
            V_var_names.append({'elements': [name], 'phases': phases, 'zero': {p: False for p in phases}})
            V_idx += 1
            V_idx_p += len(phases)

        # Add constraints. Substitute, replace and reduce decision variables.
        constraints = {"line": [], "transformer": [], "KCL": [], "KCL_internal": []}
        # Constraint: Ohm's law. Iterate over edges.
        for name, edge in edges.items():
            if edge['element_type'] in ('Line', 'Transformer'):
                zero_impedance = False
                if edge['element_type'] == 'Line':
                    T_matrix, zero_impedance = topology.line_Y_matrix(
                        topology.get_element(edge, net_data), 
                        returned_matrix='T', 
                        print_info=print_info,
                        flip_tbus_current_direction=True,
                    )
                    if zero_impedance:
                        # Source and target node voltages equal
                        merge_variables(V_names, V_var_names, edge['source'], edge['target'])
                        # Merge edge tbus current (and its equivalent currents) into fbus current.
                        merge_variables(I_names, I_var_names, f"{name}-fbus", f"{name}-tbus", equal_neg=False)
                    else:
                        constraints["line"].append((edge['source'], edge['target'], f"{name}-fbus", f"{name}-tbus", T_matrix))
                else:
                    transformer_element = topology.get_element(edge, net_data)
                    if transformer_element['transformer_type'] not in ('3PDYG', '3PDY', '3PYYG', '3PYY'):
                        raise RuntimeError(f"Unknown transformer type: {transformer_element['transformer_type']}")
                    Y_matrix = topology.transformer_Y_matrix(
                        transformer_element, 
                        returned_matrix='Y',
                        flip_tbus_current_direction=True,
                    )
                    constraints["transformer"].append((edge['source'], edge['target'], f"{name}-fbus", f"{name}-tbus", Y_matrix))
            elif 'OpenClose' in self.element_parents[edge['element_type']]:
                edge_element = topology.get_element(edge, net_data)
                tbus_el = [
                    el for el in edge_element['tbus']
                    if ('status' in el) and (el['name'] in edge['target'])
                ]
                assert len(tbus_el) == 1, f"Error in tbus element name in {edge_element}"
                # If switch status is open, current is strictly zero
                if tbus_el[0]['status'] == 'NO':
                    I_var_names[I_names[f"{name}-fbus"]['idx']]['zero'] = {p: True for p in I_var_names[I_names[f"{name}-fbus"]['idx']]['zero']}
                    I_var_names[I_names[f"{name}-tbus"]['idx']]['zero'] = {p: True for p in I_var_names[I_names[f"{name}-tbus"]['idx']]['zero']}
                # If switch status is closed, fbus and tbus currents are equal. Voltages are equal.
                elif tbus_el[0]['status'] == 'NC':
                    # Source and target node voltages equal
                    merge_variables(V_names, V_var_names, edge['source'], edge['target'])
                    # Merge edge tbus current (and its equivalent currents) into fbus current.
                    merge_variables(I_names, I_var_names, f"{name}-fbus", f"{name}-tbus", equal_neg=False)
                else:
                    raise RuntimeError(f"Unknown switch status: {tbus_el[0]['status']}")
            else:
                raise ValueError(f"Unknown element type: {edge['element_type']}")
        # Constraints: KCL (sum of currents at each node is zero)
        for name, node in nodes.items():
            if not len(node['edges']):
                if print_info: print(f"[Warning] Disconnected node with injections but no connected edge: {name}")
            # Get non-zero connected edges and injections and their directions
            conn_edges = []
            for edge_name in node['edges']:
                sign = topology.edge_direction(edges[edge_name], name)
                elem = f"{edge_name}-{'tbus' if sign == +1 else 'fbus'}"
                if not all(I_var_names[I_names[elem]['idx']]['zero'].values()):
                    conn_edges.append((elem, sign))
            conn_injs = [
                (dec_name, +1) for dec_name in node['decorators']
                if ('ACInjection' in self.element_parents[decorators[dec_name]['element_type']])
                and (not all(I_var_names[I_names[dec_name]['idx']]['zero'].values()))
            ]
            conn_elements = conn_edges + conn_injs
            # Special case: For chains, replace and reduce the number of decision variables
            if len(conn_elements) == 2:
                (elem_a, sign_a), (elem_b, sign_b) = conn_elements[0], conn_elements[1]
                merge_variables(I_names, I_var_names, elem_a, elem_b, equal_neg=(sign_a==sign_b))
            # General case: KCL with multiple elements
            elif len(conn_elements) > 2:
                constraints['KCL'].append(conn_elements)
            # Unusual case: Node with only one edge or injection
            elif len(conn_elements) == 1:
                if len(conn_edges):
                    if print_info: print(f"[Warning] Terminal node with only 1 connected edge: {name}")
                I_var_names[I_names[conn_elements[0][0]]['idx']]['zero'] = {p: True for p in I_var_names[I_names[conn_elements[0][0]]['idx']]['zero']}
            else:
                if print_info: print(f"[Warning] Disconnected node with no connected edge/injection: {name}")
        # Constraint: neutral grounding and neutral element
        for name, edge in edges.items():
            el = topology.get_element(edge, net_data)
            if el['phases'] == 'abc':
                I_var_names[I_names[f"{name}-fbus"]['idx']]['zero']['n'] = True
                I_var_names[I_names[f"{name}-tbus"]['idx']]['zero']['n'] = True
        # Constraints: Emergency generators are always off
        for name, decorator in decorators.items():
            if "Generator" in self.element_parents[decorator['element_type']]: 
                # and decorator['status'] == 'OFF':
                I_var_names[I_names[name]['idx']]['zero'] = {p: True for p in I_var_names[I_names[name]['idx']]['zero']}

        # Re-index I_names, V_names by removing reduced variables
        I_removed_cnt, V_removed_cnt, I_cnt, V_cnt = [], [], 0, 0
        for info in I_var_names:
            if not info: I_cnt += 1
            I_removed_cnt.append(I_cnt)
        for info in V_var_names:
            if not info: V_cnt += 1
            V_removed_cnt.append(V_cnt)
        for info in I_names.values():
            info['idx'] -= I_removed_cnt[info['idx']]
        for info in V_names.values():
            info['idx'] -= V_removed_cnt[info['idx']]
        # Remove reduced variables from I/V_var_names, then declare and add index into I, V array.
        I_var_names = [info for info in I_var_names if info]
        V_var_names = [info for info in V_var_names if info]
        I_var_idx, V_var_idx, I_var_idx_p, V_var_idx_p = 0, 0, 0, 0
        for info in I_var_names:
            # Note: Consider per-phase here. This will save some memory but not necessarily speed.
            #   The single-phase zero-constrained I, V optimization variables are never accessed.
            if all(info['zero'].values()): continue
            info['idx'] = I_var_idx_p
            I_var_idx += 1
            # Note: similarly, need to change this if we consider per-phase zero constraints.
            I_var_idx_p += len(info['phases'])
        for info in V_var_names:
            # Note: same here. e.g. split-phase grounding.
            if all(info['zero'].values()): continue
            info['idx'] = V_var_idx_p
            V_var_idx += 1
            V_var_idx_p += len(info['phases'])
        I = cp.Variable((I_var_idx_p, 1), complex=True)
        V = cp.Variable((V_var_idx_p, 1), complex=True)
        if print_info: print(f"Number of current and voltage decision variables (elements/per-phase): {I_var_idx}/{I_var_idx_p}, {V_var_idx}/{V_var_idx_p}")
        if print_info: print(f"Number of current and voltage elements total (elements/per-phase): {I_idx}/{I_idx_p}, {V_idx}/{V_idx_p}")

        # Instantiate constraints
        optim_constraints = []
        # Ohm's law (I=YV) constraints
        for V1, V2, I1, I2, T_matrix in constraints["line"]:
            # Get decision variable indices based on name using I_names and I_var_names
            # Also handles element(s) constrained to be zero and thus without key 'idx' in I_var_names.
            V1 = name2arr(V, V_names, V_var_names, V1)
            I1 = name2arr(I, I_names, I_var_names, I1, sign=True)
            V2 = name2arr(V, V_names, V_var_names, V2)
            I2 = name2arr(I, I_names, I_var_names, I2, sign=True)
            optim_constraints.append(cp.hstack([V1, I1]) == T_matrix @ cp.hstack([V2, I2]))
        for V1, V2, I1, I2, Y_matrix in constraints["transformer"]:
            V1 = name2arr(V, V_names, V_var_names, V1)
            I1 = name2arr(I, I_names, I_var_names, I1, sign=True)
            V2 = name2arr(V, V_names, V_var_names, V2)
            I2 = name2arr(I, I_names, I_var_names, I2, sign=True)
            # Assumes zero shunt. Can be added later.
            optim_constraints.append(cp.hstack([I1, I2]) == Y_matrix @ cp.hstack([V1, V2]))
        # KCL constraints at all nodes
        for element_list in constraints["KCL"]:
            stack = []
            for element, KCL_sign in element_list:
                idx, sign = name2idx(I_names, I_var_names, element, sign=True)
                # If the element is a zero-constrained element without key 'idx', simply skip
                if any([i is not None for i in idx]):
                    stack.append(make_arr(I, idx, sign) * KCL_sign)
            # Be careful with the stack dimension. KCL is per-phase. Output shape is (4,).
            if stack:
                optim_constraints.append(cp.sum(cp.vstack(stack), axis=0) == 0)
        # Constraint: internal KCL constraints (sum of phase currents equal zero)
        #   This is when we have a Delta or Wye connected device (e.g. transformer, load)
        kcl_internal_constraints = set()
        for name, edge in edges.items():
            if "Transformer" not in self.element_parents[edge['element_type']]: continue
            el = topology.get_element(edge, net_data)
            info = I_var_names[I_names[f"{name}-fbus"]['idx']]
            kcl_internal_constraints.add(tuple([
                info['idx'] + info['phases'].index(p) 
                for p in I_names[f"{name}-fbus"]['phases']
                if not info['zero'][p]
            ]))
            info = I_var_names[I_names[f"{name}-tbus"]['idx']]
            kcl_internal_constraints.add(tuple([
                info['idx'] + info['phases'].index(p) 
                for p in I_names[f"{name}-tbus"]['phases']
                if not info['zero'][p]
            ]))
        for idx in kcl_internal_constraints:
            optim_constraints.append(cp.sum(I[list(idx)]) == 0)
        
        # Convert metered timeseries into phasor and stack into arrays
        I_metered, V_metered, V_diff_metered = [], [], []
        I_metered_idx, V_metered_idx, V_diff_metered_idx = [], [], [[], []]
        I_met_cnt, V_met_cnt, V_diff_met_cnt = 0, 0, 0
        I_metered_nominal, V_metered_nominal, V_diff_metered_nominal = [], [], []
        for element_phase, df in sorted(metered_df.items()):
            element, phase = element_phase.split('.')
            if (element not in I_names) and (f"{element}-fbus" not in I_names) and (f"{element}-tbus" not in I_names) and (element not in V_names):
                if print_info: print(f"[Warning] A measured edge/node is not in the network: {element}.")
                continue
            phasor_arr = phasor_utils.polar2rectangular(df['phase'], df['rms'])
            if unit_lookup[element_phase] == 'A':
                element_type = (edges[element] if element in edges else decorators[element])['element_type']
                # Assumption: metered edges are always pointing into the node by sign convention (hence tbus).
                edge_or_inj = "PowerTransfer" in self.element_parents[element_type]
                idx = I_names[f"{element}-tbus" if edge_or_inj else element]['idx']
                if all(I_var_names[idx]['zero'].values()):
                    if print_info: print(f"[Warning] The metered element {element_phase} has zero current by network topology.")
                    continue
                I_metered_idx.append(I_var_names[idx]['idx'] + I_var_names[idx]['phases'].index(phase))
                I_metered.append(phasor_arr * I_names[f"{element}-tbus" if edge_or_inj else element]['sign'])
                I_metered_nominal.append(nominal_values[element])
                utils.insert_dict(I_var_names, [idx, 'meter', phase], I_met_cnt)
                I_met_cnt += 1
            elif unit_lookup[element_phase] == 'V':
                phase = phase.replace('ag', 'a').replace('bg', 'b').replace('cg', 'c').replace('ng', 'n')
                idx = V_names[element]['idx']
                # Seperate treatment for line-to-line and line-to-ground measurements
                if all(V_var_names[idx]['zero'].values()):
                    if print_info: print(f"[Warning] The metered element {element_phase} has zero voltage by network topology.")
                    continue
                if len(phase) == 1:
                    V_metered.append(phasor_arr)
                    V_metered_idx.append(V_var_names[idx]['idx'] + V_names[element]['phases'].index(phase))
                    V_metered_nominal.append(nominal_values[element])
                    utils.insert_dict(V_var_names, [idx, 'meter', phase], V_met_cnt)
                    V_met_cnt += 1
                elif len(phase) == 2:
                    V_diff_metered.append(phasor_arr)
                    element_idx, phases = V_var_names[idx]['idx'], V_var_names[idx]['phases']
                    V_diff_metered_idx[0].append(element_idx + phases.index(phase[0]))
                    V_diff_metered_idx[1].append(element_idx + phases.index(phase[1]))
                    # Note: This is not exactly right. We need to distinguish between line-to-line (* sqrt(3)) and other types of measurements. 
                    V_diff_metered_nominal.append(nominal_values[element])
                    utils.insert_dict(V_var_names, [idx, 'meter', phase], V_diff_met_cnt)
                    V_diff_met_cnt += 1
                else:
                    raise RuntimeError(f"Phase {phase} cannot be parsed for {element}")
            else:
                raise RuntimeError(f"Unknown unit: {unit_lookup[element_phase]}")
        I_metered = np.stack(I_metered, axis=1)
        V_metered = np.stack(V_metered, axis=1)
        V_diff_metered = np.stack(V_diff_metered, axis=1) if V_diff_metered else np.array([[]], dtype=np.csingle)
        # Very useful for debugging
        # utils.save_json('temp/IV_names.json', I_names | V_names)
        # utils.save_json('temp/IV_var_names.json', [{i: e for i, e in enumerate(I_var_names)}, {i: e for i, e in enumerate(V_var_names)}])

        if print_info: print("Solving...")
        # State estimation: iterate over time
        I_sol, V_sol, residual, time_col_out = [], [], [], []
        for t in tqdm(range(len(time_col))):
            # Objective function: minimize l2 distance between metered and computed I/V values
            # Here, we normalize by CT ratings and nominal bus voltages.
            obj_fxn = cp.norm((I_metered[t][:] - I[I_metered_idx,0]) / I_metered_nominal, p=1) \
                    + cp.norm((V_metered[t][:] - V[V_metered_idx,0]) / V_metered_nominal, p=1)
            if V_diff_metered.size:
                obj_fxn += cp.norm((V_diff_metered[t][:] - (V[V_diff_metered_idx[0]] - V[V_diff_metered_idx[1]])[:,0]) / V_diff_metered_nominal, p=1)
            obj = cp.Minimize(obj_fxn)
            prob = cp.Problem(obj, optim_constraints)
            try:
                prob.solve(solver=cp.ECOS)
            except Exception as e:
                print(f"Error at {time_col[t]}: {e}")
                continue
            I_sol.append(np.array(I.value).flatten())
            V_sol.append(np.array(V.value).flatten())
            residual.append(obj.value)
            time_col_out.append(time_col[t])
        I_sol = np.stack(I_sol, axis=1)
        V_sol = np.stack(V_sol, axis=1)
        residual = np.array(residual)
        time_col_out = np.array(time_col_out)

        # Write metadata solutions lookup
        soln_lookup = {}
        for name, info in sorted(I_names.items()):
            payload = {
                'file': I_var_names[info['idx']]['elements'][0],
                'phases': info['phases'],
                'sign': info['sign'],
                "metered": 'meter' in I_var_names[info['idx']],
                'zero': I_var_names[info['idx']]['zero'],
            }
            if name.endswith('-fbus'):
                if I_names[name.replace('-fbus', '-tbus')]['idx'] == info['idx']:
                    continue
                else:
                    utils.insert_dict(soln_lookup, [name[:-5], 'fbus'], payload)
            elif name.endswith('-tbus'):
                if I_names[name.replace('-tbus', '-fbus')]['idx'] == info['idx']:
                    utils.insert_dict(soln_lookup, [name[:-5]], payload)
                else:
                    utils.insert_dict(soln_lookup, [name[:-5], 'tbus'], payload)
            else:
                soln_lookup[name] = payload
        for name, info in sorted(V_names.items()):
            soln_lookup[name] = {
                'file': V_var_names[info['idx']]['elements'][0],
                'phases': info['phases'],
                "metered": 'meter' in V_var_names[info['idx']],
                'zero': V_var_names[info['idx']]['zero'],
            }
        for d in output_data_dir:
            utils.save_json(os.path.join(d, 'metadata', f"{topology_timestamp}.json"), soln_lookup)

        # Write result to disk
        for info in I_var_names:
            df = {'t': time_col_out}
            for p, phase in enumerate(info['phases']):
                if not info['zero'][phase]:
                    df[phase] = I_sol[info['idx'] + p]
            for d in output_data_dir:
                utils.write_csv_ts(df, os.path.join(d, info['elements'][0]), mode=write_mode)
            # If measurement exists
            if 'meter' in info:
                measurement_df = {'t': time_col}
                for p, idx in info['meter'].items():
                    measurement_df[p] = I_metered[:,idx]
                for d in output_data_dir:
                    utils.write_csv_ts(measurement_df, os.path.join(d, info['elements'][0] + "-measurement"), mode=write_mode)
        for info in V_var_names:
            df = {'t': time_col_out}
            for p, phase in enumerate(info['phases']):
                if not info['zero'][phase]:
                    df[phase] = V_sol[info['idx'] + p]
            for d in output_data_dir:
                utils.write_csv_ts(df, os.path.join(d, info['elements'][0]), mode=write_mode)
            # If measurement exists
            if 'meter' in info:
                measurement_df = {'t': time_col}
                for p, idx in info['meter'].items():
                    measurement_df[p] = (V_metered if len(phase) == 1 else V_diff_metered)[:,idx]
                for d in output_data_dir:
                    utils.write_csv_ts(measurement_df, os.path.join(d, info['elements'][0] + "-measurement"), mode=write_mode)
        for d in output_data_dir:
            utils.write_csv_ts({'t': time_col_out, 'v': residual}, os.path.join(d, 'residual'), mode=write_mode)
        last_timestamp = time_col[-1]
        if last_timestamp:
            for d in output_data_dir:
                utils.save_json(os.path.join(d, 'metadata', 't.json'), {'last_timestamp': str(last_timestamp)})

    def state_estimation(
        self, 
        datetimespan,
        network_files=None,
        network_metadata_dir=None,
        input_data_dir=None,
        output_data_dir=None,
        phase_ref=None,
        delta_t_threshold=None,
        print_info=False,
        keep_cc=None,
        prune_nodes=None, 
        prune_edges=None, 
        measured_injections={},
        algorithm='branchflow_VI',
        plot=False,
    ):
        """
        Compute currents and voltages of all nodes, edges and injections and write timeseries to files.
        Assumptions:
            - The system is over-determined. There is no unknown that cannot be solved
                (e.g. no disconnected component). If this is not the case (i.e. under-determined), 
                the variables with extra degrees of freedom will be assigned arbitrary values.
            - All edges and nodes in network_files are part of the system.
                We solve for all elements in network_files.
            - Any metered edges are pointing into the node.
            - 3-phase 4-wire for all edges, nodes and injections. This assumption can be
                removed later on by (1) implementing the corresponding Y matrices, and
                (2) remove the assert statements in I_names, V_names
            - Phase names are single-character. This is easy to relax, however.
            - Emergency generators are always off.
        Note:
            - For transformers and lines with shunt, the fbus and tbus currents are different.
            - We use the physical graph instead of the electrical graph. All edges are added.
                We then substitute and reduce variables basedon equality constriants.
        :param datetimespan: tuple of two str datetime, (start, end) or None.
            If None, the last timestamp in the metadata file is used.
        All other input arguments are optional (overrides class attributes).
        :return: None, result data is written to disk
        """
        network_files = network_files or self.network_files
        network_metadata_dir = network_metadata_dir or self.network_metadata_dir
        input_data_dir = input_data_dir or self.input_data_dir
        output_data_dir = output_data_dir or self.output_data_dir
        phase_ref = phase_ref or self.phase_ref
        delta_t_threshold = delta_t_threshold or self.delta_t_threshold
        if not type(output_data_dir) is list:
            output_data_dir = [output_data_dir]
        
        if datetimespan is None:
            metadata_t = utils.load_json(os.path.join(output_data_dir[0], 'metadata', 't.json'), raise_err=False)
            if not metadata_t:
                raise RuntimeError("No metadata found. Please specify datetimespan for the first time running this script.")
            datetimespan = (metadata_t['last_timestamp'], None)
            write_mode = 'append'
        else:
            write_mode = 'insert'

        # Outer for loop: Solve state estimation for each topology over the specified datetimespan
        if type(self.topology_changes) is np.ndarray:
            lo_ind = utils.np_searchsorted(self.topology_changes, utils.strptime_np(datetimespan[0]), mode='nearest_before')
            hi_ind = utils.np_searchsorted(self.topology_changes, utils.strptime_np(datetimespan[1]), mode='nearest_after') \
                if datetimespan[1] else len(self.topology_changes)
            topology_changes = self.topology_changes[lo_ind:hi_ind]
            if not len(topology_changes):
                raise ValueError(f"Check input timestamps: {datetimespan}.")
        else:
            topology_changes = self.topology_changes
        print(f"Distinct topologies starting at the following timestamps: {topology_changes}")
        for t_idx in range(len(topology_changes)):
            if print_info: print("Loading topology data...")
            # Start and end time for which this topology is valid
            t0 = topology_changes[t_idx] if t_idx > 0 else utils.strptime_np(datetimespan[0])
            t1 = topology_changes[t_idx + 1] if t_idx + 1 < len(topology_changes) else utils.strptime_np(datetimespan[1])

            net_data, nodes, edges, decorators, input_files, unit_lookup, nominal_values = \
                self.load_topology_data(t0, network_files, input_data_dir, print_info)
            # Load meter timeseries data
            if print_info: print("Loading meter data...")
            time_col, metered_df = phasor_utils.align_phasors(
                input_files,
                time_column_file=os.path.join(input_data_dir, 't'),
                datetimespan=(t0, t1), columns=['phase', 'rms'],
                ref=phase_ref,
                delta_t_threshold=delta_t_threshold
            )
            if not len(time_col) or (len(time_col) == 1 and (time_col[0] < t0 or time_col[0] > t1)):
                print(f"No data found in the specified datetimespan: {(t0, t1)}.")
                continue
            if print_info: print("Loaded data from:", time_col[0], time_col[-1])
            # Call main helper function
            topology_t = "1960-01-01" if (len(topology_changes) == 0) or (topology_changes[0] is None) else str(topology_changes[t_idx])
            if algorithm == 'branchflow_VI':
                self._state_estimation_BFM(net_data, nodes, edges, decorators, time_col, topology_t, metered_df, unit_lookup, nominal_values, output_data_dir, print_info, write_mode)
            elif algorithm == 'businjection_VI':
                self._state_estimation_BIM(net_data, nodes, edges, decorators, time_col, topology_t, metered_df, unit_lookup, nominal_values, output_data_dir, print_info, write_mode, plot=plot, keep_cc=keep_cc, prune_nodes=prune_nodes, prune_edges=prune_edges, measured_injections=measured_injections,)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")


"""Data loading"""


def get_last_timestamp(data_dir):
    return utils.strptime_np(utils.load_json(os.path.join(data_dir, 'metadata', 't.json'))['last_timestamp'])


def get_timeseries(element, datetimespan, df_cache, metadata_cache, data_dir, return_metadata=False):
    """
    To reduce the number of files on the disk, elements that are connected
    by zero-impedance elements share the same source data file, up to +/-
    sign difference. Use this function as a wrapper for loading timeseries files.

    :param element: str, name of element
        Note: for edge elements, do not include -fbus or -tbus suffix
    :param datetimespan: tuple of two datetime.datetime objects, see read_ts
    :param df_cache: dict, supply an empty dict {} for the first call, then it will
        be automatically populated by this helper function to reduce file I/O.
    :param metadata_cache: dict, loaded metadata result. The users of this
        function do not need to worry about the detailed implementation for loading metadata.
        For the first function call, provide an empty dictionary, and the appropriate metadata cache
        will be added in-place into the dictionary to reduce repeated computation. 
        If a new topology has been created, you must clear the cache.
    :param data_dir: str, path to data directory
    :return: One of the following, depending on the result
        - None if no timeseries is available
        - or dict, e.g. {
                't': np.ndarray, dtype=np.datetime64, time column
                'a': np.ndarray, dtype=np.csingle, per-phase value (voltage for nodes, current for edges)
                'b': np.ndarray, the same as above
                'c': np.ndarray, the same as above
            }
        - or tuple of two dict, each of which is defined as above, one for fbus current and one for tbus current
    """
    read_dtypes = {col: 'csingle' for col in ["a","b","c","n","g","a_measurement","b_measurement","c_measurement","n_measurement"]}
    def read_and_cache(df_cache, file, t_hash, datetimespan):
        if utils.dict_exists(df_cache, [file, t_hash]):
            df = df_cache[file][t_hash]
        else:
            df = utils.read_ts(os.path.join(data_dir, file), dtypes=read_dtypes, datetimespan=datetimespan)[0]
            utils.insert_dict(df_cache, [file, t_hash], copy.deepcopy(df))
        return df
    def make_zero_df(df_cache, phases, t_hash, datetimespan):
        t = read_and_cache(df_cache, 'residual', t_hash, datetimespan)['t']
        df = {p: np.zeros(len(t), dtype=np.csingle) for p in phases}
        df['t'] = t
        return df
    def sign_and_phase(df, file_sign):
        for column in list(df.keys()):
            if column == 't':
                continue
            elif column not in file_sign['phases']:
                df.pop(column)
            elif 'sign' in file_sign:
                df[column] *= file_sign['sign']
        return df

    dt0, dt1 = utils.strptime_np(datetimespan[0]), utils.strptime_np(datetimespan[1])
    if not metadata_cache:
        metadata_cache['timestamps'], metadata_cache['soln_lookup'] = [], []
        for f in sorted(os.listdir(os.path.join(data_dir, 'metadata'))):
            if f == 't.json': continue
            try:
                metadata_cache['timestamps'].append(utils.strptime_np(f[:-5]))
                metadata_cache['soln_lookup'].append(utils.load_json(os.path.join(data_dir, 'metadata', f), raise_err=True))
            except parser._parser.ParserError:
                raise RuntimeError(f"Invalid metadata file: {f}")
        metadata_cache['timestamps'] = np.array(metadata_cache['timestamps']).astype('datetime64[us]')

    # Check that requested datetimespan is within the range of available data
    i0 = utils.np_searchsorted(metadata_cache['timestamps'], dt0, mode='nearest_before')
    i1 = utils.np_searchsorted(metadata_cache['timestamps'], dt1, mode='nearest_after') \
        if datetimespan[1] else len(metadata_cache['timestamps'])

    for t_idx in range(i0, i1):
        t0 = metadata_cache['timestamps'][t_idx] if (t_idx > i0) or (dt0 is None) else utils.strptime_np(datetimespan[0])
        t1 = metadata_cache['timestamps'][t_idx + 1] if t_idx + 1 < i1 else utils.strptime_np(datetimespan[1])
        t_hash = (t0.astype(datetime).timestamp(), t1.astype(datetime).timestamp())
        soln_lookup = metadata_cache['soln_lookup'][t_idx]
        # For non-existent element, return None
        if element not in soln_lookup:
            return (None, None) if return_metadata else None
        df_list = []
        # For edge elements with different fbus and tbus currents
        if ('fbus' in soln_lookup[element]) and ('tbus' in soln_lookup[element]):
            if not df_list: df_list = ([], [])
            # Load fbus timeseries
            file_sign = soln_lookup[element]['fbus']
            if all(file_sign['zero'].values()):
                df_fbus = make_zero_df(df_cache, file_sign['phases'], t_hash, datetimespan)
            else:
                df_fbus = read_and_cache(df_cache, file_sign['file'], t_hash, (t0, t1))
                df_fbus = sign_and_phase(df_fbus, file_sign)
            # Load tbus timeseries
            file_sign = soln_lookup[element]['tbus']
            if all(file_sign['zero'].values()):
                df_tbus = make_zero_df(df_cache, file_sign['phases'], t_hash, datetimespan)
            else:
                df_tbus = read_and_cache(df_cache, file_sign['file'], t_hash, (t0, t1))
                df_tbus = sign_and_phase(df_tbus, file_sign)
            df_list[0].append(df_fbus)
            df_list[1].append(df_tbus)
        # For all other elements
        else:
            file_sign = soln_lookup[element]
            if all(file_sign['zero'].values()):
                df = make_zero_df(df_cache, file_sign['phases'], t_hash, datetimespan)
            else:
                df = read_and_cache(df_cache, file_sign['file'], t_hash, (t0, t1))
                df = sign_and_phase(df, file_sign)
            df_list.append(df)
    df_list = (utils.concatenate_df(df_list[0]), utils.concatenate_df(df_list[1])) \
        if type(df_list) is tuple else utils.concatenate_df(df_list)
    return (df_list, soln_lookup[element])if return_metadata else df_list


def print_results(output_data_dir, datetimespan, unique=True, network_files=None):
    """
    For debugging purposes. Output the results of state estimation in json file.
    :param output_data_dir: str, path to output data directory
    :param datetimespan: tuple of two datetime.datetime objects, see read_ts
    :param unique: If True, only return the elements with unique values,
        otherwise print out all elements.
    :param network_files: list of str, list of network files for which elements are to be loaded
    """
    result = {}
    if unique:
        for element in os.listdir(output_data_dir):
            if element == 'metadata': continue
            df = utils.read_ts(os.path.join(output_data_dir, element), datetimespan=datetimespan, default_dtype='csingle')[0]
            result[element] = {k: phasor_utils.rectangular2polar(df[k]) for k in df if k != 't'}
    else:
        # Load topology data
        assert network_files, "network_files must be provided if unique=False"
        net_data = topology.load_net_files(network_files)
        element_parents = utils.dict_parents(utils.load_json(FILE_PATHS["element_inheritance"], raise_err=True))
        nodes, edges, decorators = topology.net2graph(
            net_data, element_parents, fix_implicit=False, connected_edges=True, print_warning=True
        )
        result = {'edges': {}, 'nodes': {}, 'injections': {}}
        for name in edges:
            df_general_fbus, _ = utils.read_ts(os.path.join(output_data_dir, name + '-fbus'), datetimespan=datetimespan)
            df_general_tbus, _ = utils.read_ts(os.path.join(output_data_dir, name + '-tbus'), datetimespan=datetimespan)
            result['edges'][name + '-fbus'] = df_general_fbus
            result['edges'][name + '-tbus'] = df_general_tbus
        for name, dec in decorators.items():
            if 'ACInjection' not in element_parents[dec['element_type']]: continue
            df_general, _ = utils.read_ts(os.path.join(output_data_dir, name), datetimespan=datetimespan)
            result['injections'][name] = df_general
        for name in nodes:
            df_general, _ = utils.read_ts(os.path.join(output_data_dir, name), datetimespan=datetimespan)
            result['nodes'][name] = df_general
    utils.save_json(os.path.join(output_data_dir, 'check_results.json'), utils.serialize_dict(result))


def plot_results(
    data_dir, 
    datetimespan, 
    elements, 
    outdir=None, 
    plot_metered=True, 
    ext='png',
    combine_3_phase=False,
    base_figsize=(8, 3.5),
    fontsize=16,
    title=True,
    show=True,
    plot_every=1,
):
    """
    This is for the BIM implementation only.
    Currently only support voltage measurements. 3-phase only.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams["mathtext.fontset"] = 'cm'
    timefmt = mdates.DateFormatter('%M:%S')
    colors = [[1.00, 0.43, 0.12], [0.19, 0.63, 0.29], [0.14, 0.48, 0.72]]

    df_cache, metadata_cache= {}, {}
    if outdir: utils.mkdir(outdir)
    for name_full in elements:
        name, terminal = name_full, None
        if name_full.endswith('-fbus') or name_full.endswith('-tbus'):
            name, terminal = name_full[:-5], name_full[-4:]
        df, metadata = get_timeseries(name, datetimespan, df_cache, metadata_cache, data_dir=data_dir, return_metadata=True)
        if plot_every > 1: 
            interval_size = utils.determine_interval_size(df["t"])[0]
            interval_size = utils.round_time_np(interval_size, 's') * plot_every
            t0, t1 = utils.round_time_np(df['t'][0], 's'), utils.round_time_np(df['t'][-1], 's')
            new_t_col = np.arange(t0, t1, interval_size)
            df = utils.select_sample(df, new_t_col)
        if type(df) is tuple:
            assert terminal, f"Terminal (-fbus/-tbus) must be specified in elements for {name}."
            df = df[0] if terminal == 'fbus' else df[1]
            metadata = metadata[terminal]
        if df is None:
            print(f"[Warning] No data found for {name_full}")
            continue
        if combine_3_phase:
            fig, axs = plt.subplots(2, sharex='col', figsize=(base_figsize[0], 2*base_figsize[1]))
        else:
            fig, axs = plt.subplots(3, 2, figsize=(2*base_figsize[0], 3*base_figsize[1]))

        if title: 
            if len(name) < 48:
                fig.suptitle(name)
            else:
                fig.suptitle(name if len(name) < 256 else name[:256] + '...', fontsize=base_figsize[0] * 128/len(name))
        if name.endswith('-I') or (not name.startswith('bus')):
            vi, VI, unit = 'Current', 'I', 'A'
        else:
            vi, VI, unit = 'Voltage', 'V', 'V'
        mean_mag, mean_phase, plot_handles = [], [], []
        for i, p in enumerate('abc'):
            mag, angle = np.abs(df[p]), np.angle(df[p]) / np.pi * 180
            angle = phasor_utils.smooth_phase_180_oscillation(angle, buffer=30)
            rms_ax = axs[0] if combine_3_phase else axs[i, 0]
            phase_ax = axs[1] if combine_3_phase else axs[i, 1]
            s = rms_ax.scatter(df['t'], mag, marker='o', facecolors='none', s=base_figsize[1]*12, edgecolors=colors[i] if combine_3_phase else None, label=(f"Phase {p}" if combine_3_phase else "state est."))
            plot_handles.append(s)
            rms_ax.set_ylabel(f"{vi} magnitude ({unit})" if combine_3_phase else f"$|{VI}^{p}$|")
            phase_ax.scatter(df['t'], angle, marker='o', facecolors='none', s=base_figsize[1]*12, edgecolors=colors[i] if combine_3_phase else None, label=(f"Phase {p}" if combine_3_phase else "state est."))
            phase_ax.set_ylabel(f"{vi} phase ($^\circ$)" if combine_3_phase else rf"$\angle {VI}^{p}$")
            phase_ax.set_yticks((-180, -90, 0, 90, 180))
            mean_mag.append(mag.mean())
            mean_phase.append(angle.mean())
        plot_handles_metered = []
        if metadata["metered"] and plot_metered:
            metered_name = os.path.join(data_dir, metadata['file'])
            df_metered, err = utils.read_ts(metered_name + '-measurement', datetimespan)
            if plot_every > 1:
                df_metered = utils.select_sample(df_metered, new_t_col)
            if (err not in (1, 2)) and all([p in df_metered for p in 'abc']):
                for i, p in enumerate('abc'):
                    mag, angle = np.abs(df_metered[p]), np.angle(df_metered[p]) / np.pi * 180
                    s = rms_ax.scatter(df_metered['t'], mag, marker='x', s=base_figsize[1]*12, facecolors=colors[i] if combine_3_phase else None, label=(f"Phase {p}" if combine_3_phase else "measurement"))
                    plot_handles_metered.append(s)
                    phase_ax.scatter(df_metered['t'], angle, marker='x', s=base_figsize[1]*12, facecolors=colors[i] if combine_3_phase else None, label=(f"Phase {p}" if combine_3_phase else "measurement"))
                    mean_mag[i] = (mag.mean() + mean_mag[i]) / 2
                    mean_phase[i] = (angle.mean() + mean_phase[i]) / 2
            else:
                print(f"[Warning] No ground truth measurement data for {name_full}.")
        mag_delta = 0.10 * utils.list_mean(mean_mag)
        if combine_3_phase:
            if utils.list_mean(mean_mag) < 0.1:
                axs[0].set_ylim(-0.2, 1)
            axs[1].set_ylim(-180, 180)
            axs[1].set_xlabel(f"Time (minute:second)")
            axs[1].xaxis.set_major_formatter(timefmt)
            axs[0].add_artist(axs[0].legend(plot_handles, [f"Phase {p}" for p in 'ABC'], framealpha=0.95, loc='lower right', labelspacing=0.42, handletextpad=0.6, title='Estimation', title_fontproperties={'weight': 'bold'}))
            if plot_handles_metered:
                axs[0].legend(plot_handles_metered, [f"Phase {p}" for p in 'ABC'], framealpha=0.95, loc='upper right', labelspacing=0.42, handletextpad=0.6, title='Measurement', title_fontproperties={'weight': 'bold'})
            axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        else:
            for i in range(len('abc')):
                if mean_mag[i] + mag_delta < 0.1:
                    axs[i, 0].set_ylim(-0.2, 1)
                else:
                    axs[i, 0].set_ylim(mean_mag[i] - mag_delta, mean_mag[i] + mag_delta)
                axs[i, 1].set_ylim(mean_phase[i] - 15, mean_phase[i] + 15)
                if i < 2:
                    axs[i, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    axs[i, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[2, 0].set_xlabel(f"Time (minute:second)")
            axs[2, 1].set_xlabel(f"Time (minute:second)")
            axs[2, 0].xaxis.set_major_formatter(timefmt)
            axs[2, 1].xaxis.set_major_formatter(timefmt)
            axs[0, 0].legend()
        filename = name_full[:128]
        if (len(name) > 128) and (name[-2:] == '-I'):
            filename += '-I'
        fig.tight_layout()
        if outdir:
            plt.savefig(os.path.join(outdir, f"{filename}.{ext}"))
        if show: 
            plt.show()
        else:
            plt.close()
        

def compute_error(data_dir, datetimespan, zero_threshold=1e-2):
    """
    """
    files = os.listdir(data_dir)
    pairs = {}
    for f in files:
        if f.endswith('-measurement'):
            pairs[f[:-12]] = f
        elif f + '-measurement' in files:
            pairs[f] = f + '-measurement'
        else:
            pairs[f] = None
    errors, samples = {}, []
    for name, metered_name in pairs.items():
        if metered_name is None: continue
        df, err = utils.read_ts(os.path.join(data_dir, name), datetimespan)
        df_metered, err = utils.read_ts(os.path.join(data_dir, metered_name), datetimespan)
        not_nan = ~np.isnan(df['a']) & ~np.isnan(df['b']) & ~np.isnan(df['c'])
        not_large = (np.abs(df['a']) < 1e10) & (np.abs(df['b']) < 1e10) & (np.abs(df['c']) < 1e10)
        valid = not_nan & not_large
        samples.append(valid.sum())
        if valid.mean() < 0.9:
            print(f"[Warning] Too many NaN or large values for {name}: {(1-valid.mean()) * 100}%.")
        if any([p not in df_metered for p in 'abc']): continue
        errors[name] = {'mean': {}, 'var': {}, 'measured': {}}
        over_under = []
        for p in 'abc':
            errors[name]['mean'][p] = np.mean(np.abs(df[p][valid] - df_metered[p][valid]))
            errors[name]['var'][p] = np.var(np.abs(df[p][valid] - df_metered[p][valid]))
            over_under.append(np.mean(np.abs(df[p][valid]) - np.abs(df_metered[p][valid])))
            if np.mean(np.abs(df_metered[p])) < zero_threshold:
                zero = True
            else:
                errors[name]['measured'][p] = np.mean(np.abs(df_metered[p][valid]))
                zero = False
        over_under = np.mean(over_under)
        errors[name]['mean']['mean'] = np.mean([errors[name]['mean'][p] for p in 'abc'])
        errors[name]['var']['mean'] = np.mean([errors[name]['var'][p] for p in 'abc'])
        if not zero:
            errors[name]['measured']['mean'] = np.mean([errors[name]['measured'][p] for p in 'abc'])
        errors[name]['zero'] = zero
        print(name + (' (zero)' if zero else ''))
        print(f"mean: {'+' if over_under >= 0 else '-'}{errors[name]['mean']['mean']:.2f}, a: {errors[name]['mean']['a']:.2f}, b: {errors[name]['mean']['b']:.2f}, c: {errors[name]['mean']['c']:.2f}")
        print(f"variance: {'+' if over_under >= 0 else '-'}{errors[name]['var']['mean']:.2f}, a: {errors[name]['var']['a']:.2f}, b: {errors[name]['var']['b']:.2f}, c: {errors[name]['var']['c']:.2f}")
    print(f"Mean absolute error (%): {np.sum([d['mean']['mean'] for d in errors.values()]) / np.sum([d['measured']['mean'] for d in errors.values() if not d['zero']]) * 100}")
    print(f"Mean rms error (%): {np.sqrt(np.sum([d['mean']['mean']**2 for d in errors.values()])) / np.sqrt(np.sum([d['measured']['mean']**2 for d in errors.values() if not d['zero']])) * 100}")
    print(f"Mean variance (%): {np.mean([d['var']['mean'] for d in errors.values() if not d['zero']]) * 100}")
    print(f"Samples: {np.mean(samples)}")


if __name__ == "__main__":
    input_data_dir = FILE_PATHS['phasors']
    """State estimation (bus injection model: V-I)"""
    output_data_dir = 'temp/state_estimation_BIM'
    measured_injections = {
        "bus_1014|bus_1015|bus_1016|bus_1017|bus_1019|bus_1020|bus_1027|bus_1029|bus_1030|bus_1031|bus_1032|bus_1033|bus_1037|bus_1038|bus_1053|bus_1199": 
            [{"element": "line_381", "sign": 1}],
        "bus_1118|bus_1119|bus_1120|bus_1127": 
            [{"element": "cb_146", "sign": -1}],
        "bus_1082|bus_1083|bus_1084": 
            [{"element": "cb_137", "sign": -1}],
        "bus_1097|bus_1098|bus_1099": 
            [{"element": "cb_140", "sign": -1}],
        "bus_1106|bus_1107|bus_1108": 
            [{"element": "cb_143", "sign": -1}],
        "bus_1069|bus_1070|bus_1071|bus_1078": 
            [{"element": "cb_134", "sign": -1}],
        # Zero-injection buses below
        "bus_1034|bus_1041|bus_1042|bus_1043|bus_1044|bus_1045|bus_1046": [],
        "bus_1066|bus_1067|bus_1068": [],
        "bus_1103|bus_1104|bus_1105": [],
        "bus_1091|bus_1092|bus_1093": [],
        "bus_1054|bus_1055|bus_1056|bus_1057|bus_1058|bus_1059|bus_1079|bus_1080|bus_1081": [],
        "bus_1115|bus_1116|bus_1117": [],
    }
    datetimespan = ('2024-11-14T07:00:00', '2024-11-14T07:05:00')
    state_estimator = StateEstimator(
        network_files=[os.path.join(FILE_PATHS['net_files'], 'circuit3')], 
        input_data_dir=input_data_dir,
        output_data_dir=output_data_dir,
        phase_ref='bus_1038.ag',
        delta_t_threshold=1.0,
    )
    state_estimator.state_estimation(
        datetimespan, 
        keep_cc=[0],
        prune_nodes=["bus_1130|bus_1142|bus_1144|bus_1146|bus_1147"],
        prune_edges=[],
        measured_injections=measured_injections,
        algorithm='businjection_VI',
        print_info=True,
        plot=True,
    )
    elements_to_plot = ['bus_1033', 'bus_1034', 'bus_1118', 'bus_1082', 'bus_1097', 'bus_1106', 'bus_1069']
    elements_to_plot += ['bus_1014|bus_1015|bus_1016|bus_1017|bus_1019|bus_1020|bus_1027|bus_1029|bus_1030|bus_1031|bus_1032|bus_1033|bus_1037|bus_1038|bus_1053|bus_1199-I', 'bus_1118|bus_1119|bus_1120|bus_1127-I', 'bus_1082|bus_1083|bus_1084-I', 'bus_1097|bus_1098|bus_1099-I', 'bus_1106|bus_1107|bus_1108-I', 'bus_1069|bus_1070|bus_1071|bus_1078-I']
    elements_to_plot += ['bus_1034|bus_1041|bus_1042|bus_1043|bus_1044|bus_1045|bus_1046-I', 'bus_1066|bus_1067|bus_1068-I', 'bus_1103|bus_1104|bus_1105-I', 'bus_1091|bus_1092|bus_1093-I', 'bus_1054|bus_1055|bus_1056|bus_1057|bus_1058|bus_1059|bus_1079|bus_1080|bus_1081-I', 'bus_1115|bus_1116|bus_1117-I']
    outdir = 'temp/state_estimation_plots/results_BIM'
    plot_results(output_data_dir, datetimespan, elements_to_plot, outdir=outdir, ext='png', combine_3_phase=True, show=False)
    print_results(output_data_dir, datetimespan)
    compute_error(output_data_dir, datetimespan)

    """State estimation (branch flow model: V-I)"""
    # output_data_dir = 'temp/state_estimation_BFM'
    # datetimespan = ('2024-11-14T07:00:00', '2024-11-14T07:05:00')
    # state_estimator = StateEstimator(
    #     network_files=[os.path.join(FILE_PATHS['net_files'], 'circuit3')], 
    #     input_data_dir=input_data_dir,
    #     output_data_dir=output_data_dir,
    #     phase_ref='bus_1038.ag',
    #     delta_t_threshold=1.0,
    # )
    # state_estimator.state_estimation(datetimespan, print_info=False)
    # elements_to_plot = ['bus_1033', 'bus_1034', 'bus_1118', 'bus_1082', 'bus_1097', 'bus_1106', 'bus_1069']
    # elements_to_plot += ['line_381', 'line_383', "line_431", "cb_134", "cb_151-fbus"]
    # outdir = 'temp/state_estimation_plots/results_BFM'
    # plot_results(output_data_dir, datetimespan, elements_to_plot, outdir=outdir, ext='png', combine_3_phase=True, show=False)
    # print_results(output_data_dir, datetimespan)
    # compute_error(output_data_dir, datetimespan)
