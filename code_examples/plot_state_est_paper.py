from state_estimation_phasor import *

if __name__ == "__main__":
    """State estimation (bus injection model: V-I)"""
    output_data_dir = 'temp/state_estimation_BIM'
    measured_injections = {
        "bus_1027|bus_1029|bus_1031|bus_1019|bus_1030|bus_1199|bus_1015|bus_1014|bus_1016|bus_1032|bus_1053|bus_1037|bus_1017|bus_1020|bus_1038|bus_1033": 
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
    }
    datetimespan = ('2024-11-14T16:00:00', '2024-11-14T16:05:00')
    state_estimator = StateEstimator(
        network_files=[os.path.join(FILE_PATHS['net_files'], 'circuit3')], 
        input_data_dir=FILE_PATHS['phasors'],
        output_data_dir=output_data_dir,
        phase_ref='bus_1038.ag',
        delta_t_threshold=1.0,
    )
    state_estimator.state_estimation(
        datetimespan, 
        print_info=False,
        keep_cc=[0],
        prune_nodes=["bus_1142|bus_1144|bus_1146|bus_1147|bus_1130"],
        prune_edges=[],
        measured_injections=measured_injections,
        algorithm='businjection_VI',
    )
    # elements_to_plot = ['bus_1033']
    # elements_to_plot += ["bus_1027|bus_1029|bus_1031|bus_1019|bus_1030|bus_1199|bus_1015|bus_1014|bus_1016|bus_1032|bus_1053|bus_1037|bus_1017|bus_1020|bus_1038|bus_1033-I"]
    elements_to_plot = ['bus_1033', 'bus_1034', 'bus_1118', 'bus_1082', 'bus_1097', 'bus_1106', 'bus_1069']
    elements_to_plot += ["bus_1027|bus_1029|bus_1031|bus_1019|bus_1030|bus_1199|bus_1015|bus_1014|bus_1016|bus_1032|bus_1053|bus_1037|bus_1017|bus_1020|bus_1038|bus_1033-I", "bus_1118|bus_1119|bus_1120|bus_1127-I", "bus_1082|bus_1083|bus_1084-I", "bus_1097|bus_1098|bus_1099-I", "bus_1106|bus_1107|bus_1108-I", "bus_1069|bus_1070|bus_1071|bus_1078-I"]
    outdir = 'temp/state_estimation_plots/results_BIM'
    plot_results(output_data_dir, datetimespan, '2024-11-13T23:30:42.000000.json', elements_to_plot, outdir=outdir, ext='pdf', combine_3_phase=True, show=False, title=False)
    plot_results(output_data_dir, datetimespan, '2024-11-13T23:30:42.000000.json', elements_to_plot, outdir=outdir, ext='svg', combine_3_phase=True, show=False, title=False)
