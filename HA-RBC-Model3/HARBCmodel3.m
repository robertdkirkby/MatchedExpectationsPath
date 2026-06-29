% Heterogeneous-Agent Real Business Cycle Model 3: Entrepreneurial-Choice
% Essentially the model of Kitao (2008) with aggregate shocks
% Solution uses the matched-expectation path method of Hanbaek Lee

% assets: assets
% e: entrepreneur/worker
% eta: labor productivity
% theta: entrepreneurial ability
% A lot of the following is copy-pasted from the Kitao (2008) example codes.
% See those for more explanation of how the setup is done

% Model has two endogenous states and two exogenous markov states (you can change n_asset, but the rest are hardcoded)
n_asset=1001; % assets [also using vfoptions.ngridinterp]
n_e=2; % 1 is entrepreneur, 0 is worker
n_eta=5; % Kitao (2008) uses 5
n_theta=4; % Kitao (2004) uses 4

n_d=0;
n_a=[n_asset,n_e];
n_z=[n_eta,n_theta]; % idiosyncratic shocks
n_S=2; % aggregate shocks: recession and boom

T=1000; % number of time periods used in the matched-expectations path. 
% With T=25 can run, but with T=50 it does not, so need to divide small enough to run
recursiveeqmoptions.divideT=40;

recursiveeqmoptions.tolerance=3*10^(-4); % get to this, but cannot quite hit 1e-4, solution seems stable though and the 99.9th percentile of the price updates is under the 1e-4

% Note: The changes from Kitao are putting S into the Corporate sector
% Cobb-Douglas production function (appears in the general eqm eqns), 
% and into the Entrepreneurial sector (replace all instances of 'theta'
% with 'S*theta').

%% Parameters

Params.sigma=2; % CES utility
Params.beta=0.9428; % time preference

% Corporate sector
Params.Z=1; % Technology level (normalized to 1) (this is never actually used, Kitao (2008) calls it A but I want to use A for aggregate assets)
Params.alpha=0.36; % Cobb-Douglas prodn fn, capital share
Params.delta=0.06; % depreciation rate

% Non-corporate sector
Params.upsilon=0.88;
Params.upsilon1=Params.alpha*Params.upsilon; % pg 50 of Kitao (2008)
Params.upsilon2=(1-Params.alpha)*Params.upsilon; % pg 50 of Kitao (2008)
% the way upsilon1 and upsilon2 two are done is so that capital has same
% 'importance' in non-corporate sector as it does in corporate sector

% Borrowing
Params.phi=0.05; % additional cost of borrowing for non-corporate sector
Params.d=0.5; % max borrowing leverage

% Government
Params.tau_a0=0.258; % Three parameters for non-linear income tax schedule
Params.tau_a1=0.768;
Params.tau_a2=0.438;
Params.tau_I=0.0316; % Proportional tax on income
Params.tau_c=0.0567; % consumption tax

% Initial guess for interest rates
Params.r=0.034; % in the ballpark of 1/beta -1, which is what r would be in a complete markets model
Params.w=1.35;
Params.G=0.41;


%% Aggregate Shocks
% The aggregate shocks take two values
S_grid=[0.99; 1.01];
pi_S =[0.8750, 0.1250; 0.1250,0.8750];
% Note: this is same S_grid and pi_S as KS1998

% Give the aggregate shock a name (you must use this same name when inputting S to ReturnFn, FnsToEvaluate and GeneralEqmEqns, and it will be used to name some outputs)
AggShockNames={'S'};

%% Set up the exogenous idiosyncratic shock processes

% mu: labor productivity (from Kitao (2008), Appendix B)
eta_grid=[0.646; 0.798; 0.966; 1.169; 1.444];
pi_eta=[0.731, 0.253, 0.016, 0.000, 0.000; 0.192, 0.555, 0.236, 0.017, 0.000; 0.011, 0.222, 0.533, 0.222, 0.011; 0.000, 0.017, 0.236, 0.555, 0.192; 0.000, 0.000, 0.016, 0.253, 0.731];
% Note: third row of pi_eta actually sums to 0.999, so need to normalize it to 1
pi_eta=pi_eta./sum(pi_eta,2);

% theta: entrepreneurial ability (from Kitao (2008), Appendix B)
theta_grid=[0.000; 0.706; 1.470; 2.234];
pi_theta=[0.780, 0.220, 0.000, 0.000; 0.430, 0.420, 0.150, 0.000; 0.000, 0.430, 0.420, 0.150; 0.000, 0.000, 0.220, 0.780];

%% Grids
% Set grid for asset holdings
assetmax=110.3448; % look odd, just want to use identical to the Kitao (2008) replication
asset_grid=assetmax*(linspace(0,1,n_asset).^3)'; % linspace ^3 puts more points near zero, where the curvature of value and policy functions is higher and where model spends more time

e_grid=[0;1]; % 1 is entrepreneur, 0 is worker


%% Get into form for VFI toolkit
d_grid=[];
a_grid=[asset_grid; e_grid];
z_grid=[eta_grid; theta_grid];
pi_z=kron(pi_theta, pi_eta); % in reverse order

%%
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime,eprime,a,e,eta,theta,S,sigma,d,upsilon1,upsilon2,r,w,delta,phi,tau_a0, tau_a1, tau_a2, tau_I, tau_c)...
    Kitao2008_ReturnFn(aprime,eprime,a,e,eta,theta,S,sigma,d,upsilon1,upsilon2,r,w,delta,phi,tau_a0, tau_a1, tau_a2, tau_I, tau_c);
% The first inputs must be: decision variables, next period endogenous state, endogenous state, exogenous state. Followed by any parameters


%% Aggregates

% Create functions to be evaluated
FnsToEvaluate.K_noncorp = @(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi) Kitao2008_kFn(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi); % Assets used in non-corporate sector (=entrepreneurs)
FnsToEvaluate.A = @(aprime,eprime,a,e,eta,theta) a; % Total assets of households (workers and entrepreneurs)
FnsToEvaluate.N_noncorp = @(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi) Kitao2008_nFn(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi); % Labor used in non-corporate sector (=entrepreneurs)
FnsToEvaluate.L = @(aprime,eprime,a,e,eta,theta) eta; % Total labor supply
FnsToEvaluate.TaxRevenue = @(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi,tau_a0, tau_a1, tau_a2, tau_I, tau_c)...
    Kitao2008_TaxFn(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi,tau_a0, tau_a1, tau_a2, tau_I, tau_c); % Tax Revenue


%% General equilbrium
GEPriceParamNames={'r','w','G'}; % No need to actually have w here (because it is Cobb-Douglas prodn fn in corporate sector, and we can therefore derive a relation between w and r that must hold in stationary eqm (but not in transition)

GeneralEqmEqns.CapitalMarket = @(r,K_noncorp,A,N_noncorp,L,alpha,delta,S) r-(alpha*S*((A-K_noncorp)^(alpha-1))*((L-N_noncorp)^(1-alpha))-delta); %The requirement that the interest rate corresponds to the marginal product of capital in corporate sector
GeneralEqmEqns.LaborMarket = @(w,K_noncorp,A,N_noncorp,L,alpha,S) w-(1-alpha)*S*((A-K_noncorp)^(alpha))*((L-N_noncorp)^(-alpha)); %The requirement that the interest rate corresponds to the marginal product of capital in corporate sector
GeneralEqmEqns.GovBudget = @(G,TaxRevenue) G-TaxRevenue; %Government runs balanced budget
% % When first solving this using just these GE conditions it was giving me a solution in which L<N_noncorp.
% % So I added the following which is essentially a penalty term on L<N_noncorp.
% GeneralEqmEqns.LaborPenalty = @(L,N_noncorp) 10*abs(L-N_noncorp)*(L<=N_noncorp+0.03); % Add a penalty whenever L<N_noncorp+0.03 (seems like any solution should have at least 0.03 labor in the corporate sector)


%% Use divide-and-conquer together with grid interpolation layer when solving value fn problem
vfoptions.divideandconquer=1;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=25;
simoptions.gridinterplayer=vfoptions.gridinterplayer;
simoptions.ngridinterp=vfoptions.ngridinterp;



%% Test out the stationary general eqm (tends to be a good idea to do this to make sure we have a calibration that seems reasonable, before we turn to the aggregate shocks)
% pi_z is independent of S, so easy enough

% vfoptions.divideandconquer=0;
% heteroagentoptions.verbose=1;
% heteroagentoptions.toleranceGEcondns=1e-6
% Params.S=1;
% [p_eqm,GEcondn]=HeteroAgentStationaryEqm_InfHorz(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% 
% Params.r=p_eqm.r;
% Params.w=p_eqm.w;
% Params.G=p_eqm.G;
% 
% [V,Policy]=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
% StationaryDist=StationaryDist_InfHorz(Policy,n_d,n_a,n_z,pi_z,simoptions);
% cumDist=cumsum(sum(sum(sum(StationaryDist,4),3),2));
% figure(1);
% plot(asset_grid,cumDist)
% title('cdf of assets')
% 
% temp=sum(sum(sum(StationaryDist,4),3),1);
% fprintf('Fraction that are entrepreneurs: %1.4f \n', temp(2))
% 
% vfoptions.divideandconquer=1;


%% Solve the recursive general equilibrium with aggregate shocks
% T=1000; % number of time periods used in the matched-expectations path
recursiveeqmoptions.burnin=100; % 100 is anyway the default value, just putting this so you can see how to change it

recursiveeqmoptions.verbose=2; % Give feedback (=1 gives feedback, =2 is excessive feedback mostly intended for debugging)
recursiveeqmoptions.heteroagentoptions.verbose=1; % Turn on verbose while solving for the initial guess

% We use a shooting-algorithm to update the general eqm prices, the following two lines set this up
recursiveeqmoptions.GEnewprice=3;
recursiveeqmoptions.GEnewprice3.howtoupdate={...
    'CapitalMarket','r',0,0.1;...
    'LaborMarket','w',0,0.1;...
    'GovBudget','G',0,0.1}; % same thing as when setting up a transition path, or using fminalgo=5 for the stationary general eqm

% Graphs at each iteration on the matched expectations path that show what is going on
recursiveeqmoptions.graphaggvarspath=1;
recursiveeqmoptions.graphpricepath=1;

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

