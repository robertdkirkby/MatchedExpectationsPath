% Heterogeneous-Agent Real Business Cycle Model 1: Krusell & Smith (1998) model
% Krussell & Smith (1998) - Income and Wealth Heterogeneity in the Macroeconomy
% Solution uses the matched-expectation path method of Hanbaek Lee

n_d=0;
n_a=501; % assets
n_z=2; % idiosyncratic shocks: unemployed and employed
n_S=2; % aggregate shocks: recession and boom
T=1000; % number of time periods used in the matched-expectations path. 

makeitfast=0 % with a little bit of setting the options differently we can speed things up

%% Parameters
Params.beta      = 0.99; % discount factor

Params.sigma=1; % curvature of utility fn

Params.alpha     = 0.36; % capital share in production
Params.delta     = 0.025; % depreciation rate

% Note: w*z + b*(1-z) with z=0 and 1 can be simplified to just w*z with z=b/w and z=1
% Hence we don't need a parameter b as it is just implicit in our z grid

% Initial guess for interest rates
Params.r=0.01; % in the ballpark of 1/beta -1, which is what r would be in a complete markets model

%% Grids
d_grid=[];

% Assets from 0 to 100, the .^3 means more points near zero
a_grid=100*linspace(0,1,n_a)'.^3;

%% Idiosyncratic and Aggregate Shocks
% The aggregate shocks take two values
S_grid=[0.99; 1.01];
% Give the aggregate shock a name (you must use this same name when inputting S to ReturnFn, FnsToEvaluate and GeneralEqmEqns, and it will be used to name some outputs)
AggShockNames={'S'};

% The idiosyncratic shocks take two values, and their transition
% probabilities are determined jointly with the aggregate shock transitions
z_grid=[0.25; 1.00]; % unemployed, employed
% Note: w*z + b*(1-z) with z=0 and 1 can be simplified to just w*z with z=b/w and z=1
% So this grid is just implying that unemployment benefits are set to one-quarter of the wage

% The following is largely copy-paste from QuantEcon, it follows KS1998
% and how to create the transition probabilities for z and S.
% https://github.com/QuantEcon/krusell_smith_code/blob/master/KSfunctions.ipynb

ug=0.04; % ug - unemployment rate in good state
ub=0.1; % ub - unemployment rate in bad state
zg_ave_dur=8; % zg_ave_dur - average duration of good state
zb_ave_dur=8; % zb_ave_dur - average duration of bad state
ug_ave_dur=1.5; % ug_ave_dur - average duration of unemployment in good state
ub_ave_dur=2.5; % ub_ave_dur - average duration of unemployment in bad state
puu_rel_gb2bb=1.25; % puu_rel_gb2bb - prob. of u to u cond. on g to b relative to that of b to b
puu_rel_bg2gg=0.75; % puu_rel_bg2gg - prob. of u to u cond. on b to g relative to that of g to g

% S transition probabilties
pgg = 1-1/zg_ave_dur; % probability of remaining in good state
pbb = 1-1/zb_ave_dur; % probability of remaining in bad state
pgb = 1-pgg; % probability of changing from g to b
pbg = 1-pbb; % probability of changing from b to g
pi_S=[pbb,pbg; pgb,pgg];

% z transition probabilties
p00_gg = 1-1/ug_ave_dur; % prob. of 0 to 0 cond. on g to g
p00_bb = 1-1/ub_ave_dur; % prob. of 0 to 0 cond. on b to b
p01_gg = 1-p00_gg; % prob. of 0 to 1 cond. on g to g
p01_bb = 1-p00_bb; % prob. of 0 to 1 cond. on b to b

p00_gb=puu_rel_gb2bb*p00_bb; % prob. of 0 to 0 cond. on g to b
p00_bg=puu_rel_bg2gg*p00_gg; % prob. of 0 to 0 cond. on b to g
p01_gb=1-p00_gb; % prob. of 0 to 1 cond. on g to b
p01_bg=1-p00_bg; % prob. of 0 to 1 cond. on b to g

p10_gg=(ug - ug*p00_gg)/(1-ug); % prob. of 1 to 0 cond. on  g to g
p10_bb=(ub - ub*p00_bb)/(1-ub); % prob. of 1 to 0 cond. on b to b
p10_gb=(ub - ug*p00_gb)/(1-ug); % prob. of 1 to 0 cond. on g to b
p10_bg=(ug - ub*p00_bg)/(1-ub); % prob. of 1 to 0 cond on b to g

p11_gg= 1-p10_gg; % prob. of 1 to 1 cond. on  g to g
p11_bb= 1-p10_bb; % prob. of 1 to 1 cond. on b to b
p11_gb= 1-p10_gb; % prob. of 1 to 1 cond. on g to b
p11_bg= 1-p10_bg; % prob. of 1 to 1 cond on b to g

pi_z=zeros(n_z,n_z,n_S,n_S); % transitions depend on the aggregate shock (joint, not just dependent on current agg shock)
pi_z(:,:,1,1)=[p00_bb,p01_bb; p10_bb, p11_bb]; % In recession today and tomorrow
pi_z(:,:,1,2)=[p00_bg,p01_bg; p10_bg, p11_bg]; % In recession today and boom tomorrow
pi_z(:,:,2,1)=[p00_gb,p01_gb; p10_gb, p11_gb]; % In boom today and tomorrow
pi_z(:,:,2,2)=[p00_gg,p01_gg; p10_gg, p11_gg]; % In boom today and tomorrow

%% Return Fn
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime,a,z,r,sigma,alpha,delta)...
    HARBCmodel1_ReturnFn(aprime,a,z,r,sigma,alpha,delta);
% Note: It is possible to include S as inputs to ReturnFn, just as if they were parameters

%%
FnsToEvaluate.K=@(aprime,a,z,S) a;
FnsToEvaluate.L=@(aprime,a,z,S) z;
% Note: It is possible to include S as inputs to FnsToEvaluate, just as if they were parameters

%% General Eqm
GEPriceParamNames={'r'};

GeneralEqmEqns.capitalmarket=@(r,S,K,L,alpha,delta) r-(alpha*S*(K^(alpha-1))*(L^(1-alpha))-delta);

%% Use divide-and-conquer together with grid interpolation layer when solving value fn problem
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=50;
simoptions.gridinterplayer=vfoptions.gridinterplayer;
simoptions.ngridinterp=vfoptions.ngridinterp;
vfoptions.divideandconquer=1;

%% Solve the recursive general equilibrium with aggregate shocks
% T=1000; % number of time periods used in the matched-expectations path
recursiveeqmoptions.burnin=100; % 100 is anyway the default value, just putting this so you can see how to change it

recursiveeqmoptions.verbose=2; % Give feedback (=1 gives feedback, =2 is excessive feedback mostly intended for debugging)
recursiveeqmoptions.heteroagentoptions.verbose=1; % Turn on verbose while solving for the initial guess

% We use a shooting-algorithm to update the general eqm prices, the following two lines set this up
recursiveeqmoptions.GEnewprice=3;
recursiveeqmoptions.GEnewprice3.howtoupdate={'capitalmarket','r',0,0.1}; % same thing as when setting up a transition path, or using fminalgo=5 for the stationary general eqm

% Graphs at each iteration on the matched expectations path that show what is going on
recursiveeqmoptions.graphaggvarspath=1;
recursiveeqmoptions.graphpricepath=1;


if makeitfast==1
    heteroagentoptions.GEnewprice=recursiveeqmoptions.GEnewprice;
    heteroagentoptions.GEnewprice3.howtoupdate=recursiveeqmoptions.GEnewprice3.howtoupdate;
end

% Solves using the matched-expectations path algorithm of Hanbaek Lee
time1=datetime;
tic;
GeneralizedTransitionFn=RecursiveGeneralEqmWithAggShocks_InfHorz(T,n_d,n_a,n_z,n_S,d_grid,a_grid,z_grid,S_grid,pi_z,pi_S,ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, GEPriceParamNames, AggShockNames, recursiveeqmoptions,vfoptions,simoptions);
GEtime=toc
time2=datetime;

% Started
time1
% Finished 
time2

