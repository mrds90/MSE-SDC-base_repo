% ----------------------------------------
% Init script
% ----------------------------------------
close all;
clear all;
clc;
format compact;
graphics_toolkit gnuplot;
pkg load communications;
pkg load signal;
% ----------------------------------------


% ----------------------------------------
% Paths
% ----------------------------------------
addpath(genpath('./funcs/'));
addpath(genpath('../lib/'));
% ----------------------------------------


% ----------------------------------------
% Load System and Channel configuration
% ----------------------------------------
source('./system_parameters.m');
source('./channel_parameters.m');
% ----------------------------------------


% ----------------------------------------
% Simulation Parameters
% ----------------------------------------
N_TX    = 3; % Number of transmissions
N_ZEROS = 123; % Length of zero samples before signal
% ----------------------------------------


% ----------------------------------------
% Discrete time
% ----------------------------------------
kzeros = Ts*N_ZEROS;
khalfp = Ts*(n_fir-1)/2;
kmod   = Tsymb*(spar.n_pre + spar.n_sfd + spar.n_bytes * 8); % tiempo de transmisión (preambulo + start frame delimiter + cantidad de bits)
k0     = kzeros+khalfp;
kend   = kzeros+khalfp+kmod;
% ----------------------------------------


% ----------------------------------------
% Modulator
% ----------------------------------------
% bytes   = round(rand(1,spar.n_bytes)*255);
bytes   = 1:spar.n_bytes;
[xaux mis] = modulator(bytes,spar);
x          = [zeros(1,N_ZEROS) xaux];
d          = [mis.d];
kk         = k0:Tsymb:(kend-Tsymb);
k          = kk;
for i=1:N_TX-1
  % bytes = round(rand(1,spar.n_bytes)*255);
  bytes = bytes+1;
  [xaux mis] = modulator(bytes,spar);
  x = [x zeros(1,N_ZEROS) xaux];
  d = [d mis.d];
  k = [k kk+(kend+khalfp)*i];
end
x = [x zeros(1,N_ZEROS)];
% ----------------------------------------


% ----------------------------------------
% Channel
% ----------------------------------------
y = channel(x,cpar);
% ----------------------------------------


% ----------------------------------------
% Demodulator
% ----------------------------------------
[hat_bytes dis] = demodulator(y,spar);
% ----------------------------------------


% ----------------------------------------
% Plots
% ----------------------------------------
t = 0:Ts:Ts*(length(x)-1);
if N_TX<=10
  % Mod
  figure(); hold on;
  stem(k,d*2-1,'x','linewidth',4,'displayname','d: Deltas');
  stem(t,x,    'o','linewidth',1,'displayname','c: Canal');
  axis([0 32*Tsymb]);
  legend();
  % Channel
  figure(); hold on;
  stem(k,d*2-1,'x','linewidth',4,'displayname','d: Deltas');
  stem(t,y,    'o','linewidth',1,'displayname','y: Luego del FIR');
  axis([0 32*Tsymb]);
  legend();
  % Signal Detection
  figure(); hold on;
  plot(t,dis.y_mf       ,'linewidth', 3, 'displayname', 'y: Luego del FIR');
  plot(t,dis.y_mf_sq    ,'linewidth', 3, 'displayname', 'y^2: Elevada al cuadrado ');
  plot(t,dis.y_mf_sq_ma ,'linewidth', 3, 'displayname', 'filter(y^2): Filtrada');
  plot(t,dis.detection-2,'linewidth', 4, 'displayname', 'Detección de señal');
  legend();
  % PLL 1
  figure(); hold on;
  plot(t,dis.y_mf_sq      ,'linewidth', 3,'-');
  plot(t,dis.pllis.phd    ,'linewidth', 3,'-','displayname','phd');
  plot(t,dis.pllis.err    ,'linewidth', 3,'-','displayname','err');
  plot(t,dis.pllis.phi_hat,'linewidth', 3,'-','displayname','phi_hat');
  plot(t,dis.pll_cos/2-1  ,'linewidth', 3,'-');
  plot(t,dis.pll_sin/2-2.5,'linewidth', 3,'-');
  plot(t,dis.pll_clk_i-1.5,'linewidth', 3,'-','linewidth',2);
  plot(t,dis.pll_clk_q-3  ,'linewidth', 3,'-','linewidth',2);
  legend();
  % PLLt, 2
  figure(); hold on;
  plot(t,dis.detection,'linewidth',4);
  plot(t,dis.pll_cos/2-1);
  plot(t,dis.pll_sin/2-2.5);
  plot(t,dis.pll_clk_i-1.5,'linewidth',2);
  plot(t,dis.pll_clk_q-3,'linewidth',2);
  legend();
  % Synch
  figure(); hold on;
  plot(t,dis.y_mf          -7,'-o','displayname','y_{fa}');
  plot(t,dis.y_mf_pf       -7,'-x','displayname','y_{fa-sq}');
  plot(t,dis.y_mf_pf_sq    -7,'-s','displayname','y_{fa-sq-ma}');
  plot(t,dis.y_mf_pf_sq_bpf-7,'-.','displayname','y_{fa-sq-ma}');
  plot(t,dis.detection-4.5,'linewidth',2);
  stem(k, 2*ones(size(k)),'k.');
  stem(k,-5*ones(size(k)),'k.');
  plot(t,dis.pllis.phd    ,'-');
  plot(t,dis.pllis.err    ,'-');
  plot(t,dis.pllis.phi_hat,'-');
  plot(t,dis.flank_qp-3   ,'-o','linewidth',2,'displayname','QP');
  plot(t,dis.flank_qn-3   ,'-o','linewidth',2,'displayname','QN');
  plot(t,dis.flank_ip-3   ,'-o','linewidth',2,'displayname','IP');
  plot(t,dis.flank_in-3   ,'-o','linewidth',2,'displayname','IN');
  axis([0 max(t) -5 2]);
  legend('show');
else
  % figure(); hold on;
  % hist(x,63);
end
% ----------------------------------------

