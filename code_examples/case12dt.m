function mpc = case12dt
%CASE12DT  Power flow data for the Netlab 12 bus distribution system.
%All data is obtained from a real-world circuit.

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 1;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [ %% (Pd and Qd are specified in kW & kVAr here, converted to MW & MVAr below)
	1033	3	0		0		0	0	1	1	0	16.5	1	1	1;
	1034	1	0		0		0	0	1	1	0	2.4		1	1.1	0.9;
	1068	1	0		0		0	0	1	1	0	2.4		1	1.1	0.9;
	1081	1	0		0		0	0	1	1	0	2.4		1	1.1	0.9;
	1093	1	0		0		0	0	1	1	0	2.4		1	1.1	0.9;
	1105	1	0		0		0	0	1	1	0	2.4		1	1.1	0.9;
	1117	1	0		0		0	0	1	1	0	2.4		1	1.1	0.9;
	1069	1	70.5	42.4	0	0	1	1	0	0.48	1	1.1	0.9;
	1082	1	107.6	34.8	0	0	1	1	0	0.48	1	1.1	0.9;
	1097	1	38.4	38.5	0	0	1	1	0	0.48	1	1.1	0.9;
	1106	1	9.1		-2.9	0	0	1	1	0	0.48	1	1.1	0.9;
	1118	1	115.5	56.0	0	0	1	1	0	0.48	1	1.1	0.9;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
    1033    0	0	10	-10	1	100	1	10	0	0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [  %% (r and x specified in ohms here, converted to p.u. below)
	1033	1034	0.0093	0.1158	0	0	0	0	1	0	1	-360	360;
	1034	1081	0.0376	0.2479	0	0	0	0	0	0	1	-360	360;
	1034	1117	0.0028	0.0187	0	0	0	0	0	0	1	-360	360;
	1081	1068	0.0198	0.1306	0	0	0	0	0	0	1	-360	360;
	1081	1093	0.0142	0.0933	0	0	0	0	0	0	1	-360	360;
	1081	1105	0.0170	0.1120	0	0	0	0	0	0	1	-360	360;
	1081	1082	0.0011	0.0132	0	0	0	0	1	0	1	-360	360;
	1117	1118	0.0007	0.0090	0	0	0	0	1	0	1	-360	360;
	1068	1069	0.0007	0.0084	0	0	0	0	1	0	1	-360	360;
	1093	1097	0.0013	0.0168	0	0	0	0	1	0	1	-360	360;
	1105	1106	0.0012	0.0149	0	0	0	0	1	0	1	-360	360;
];

%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	0	0	3	0	20	0;
];


%% convert branch impedances from Ohms to p.u.
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
Vbase = mpc.bus(1, BASE_KV) * 1e3;      %% in Volts
Sbase = mpc.baseMVA * 1e6;              %% in VA
mpc.branch(:, [BR_R BR_X]) = mpc.branch(:, [BR_R BR_X]) / (Vbase^2 / Sbase);

%% convert loads from kW to MW
mpc.bus(:, [PD, QD]) = mpc.bus(:, [PD, QD]) / 1e3;
