% Heterogeneous-Agent Real Business Cycle Model
% Uncertainty Shocks, which determine the volatility of both the aggregate
% TFP shocks and the idiosyncratic productivity shocks.

n_d=0; % labor supply
n_a=601; % assets
n_z=4; % idiosyncratic shocks: idiosyncratic labor productivity
n_S=[5,2]; % aggregate shocks: TFP, and uncertainty shock [S_TFP and S_Uncert]
T=10000; % number of time periods used in the matched-expectations path. 
        % Because there are more aggregate shocks, we likely need more time periods
recursiveeqmoptions.divideT=15; % T is big enough that run out of GPU memory, so we divide T into 2 chunks when doing VFI which reduces memory use at a moderate runtime cost

recursiveeqmoptions.tolerance=3*10^(-4); % get to this, but cannot quite hit 1e-4, solution seems stable though and the 99.9th percentile of the price updates is under the 1e-4

separate_rw=0 % =0: solve for r using capital market, then hardcode eqn for w
              % =1: solve for r&w using capital market and labor market
% works for both 0 and 1
% with separate_rw, when I had weights of 0.1 on both it would fail, but
% seems like with a lower weight on w it is okay

%% Parameters
Params.beta = 0.94; % discount factor

Params.sigma=2; % curvature of utility fn in consumption
Params.sigma_l=2; % curvature of utility fn in leisure
Params.phi=3; % relative importance of leisure

Params.alpha     = 0.36; % capital share in production
Params.delta     = 0.025; % depreciation rate

% All three shocks have mean of 1, and are (log) AR1
% Aggregate shock: Uncertainty shock
Params.rhoS2=0.5;
Params.sigma_S2epsilon=0.1;
% Aggregate shock: TFP
Params.rhoS1=0.7;
Params.sigma_S1epsilon=0.1; % times S_Uncert
% Idiosyncratic shock: effective labor productivity units
Params.rho_z=0.9;
Params.sigma_zepsilon=0.1; % times S_Uncert

% Initial guess for interest rates
Params.r=0.05;
if separate_rw==1
    Params.w=1.3;
end


%% Grids
d_grid=linspace(0,1,n_d)';

% Assets from 0 to 50, the .^3 means more points near zero
a_grid=50*linspace(0,1,n_a)'.^3;

%% Idiosyncratic and Aggregate Shocks
% Give the aggregate shock a name (you must use this same name when inputting S to ReturnFn, FnsToEvaluate and GeneralEqmEqns, and it will be used to name some outputs)
AggShockNames={'S_TFP', 'S_Uncert'};

pi_S=zeros(prod(n_S),prod(n_S));
S_gridvals=zeros(prod(n_S),length(n_S)); % joint-grid, because values of the S1 grid depend on S2

% S_Uncert is an AR(1) process
if n_S(2)==1 % Just so you can easily turn of uncertainty shocks to compare model results
    S_Uncert_grid=1;
    pi_S_Uncert=1;
elseif n_S(2)==2
    [S_Uncert_grid, pi_S_Uncert]=discretizeAR1_Tauchen(0, Params.rhoS2, Params.sigma_S2epsilon, n_S(2),0.5);
    S_Uncert_grid=exp(S_Uncert_grid);
    [meanS_Uncert,~,~,~]=MarkovChainMoments(S_Uncert_grid,pi_S_Uncert);
    S_Uncert_grid=S_Uncert_grid/meanS_Uncert; % make sure E[S2]=1
else % Farmer-Toda requires using a least three points
    [S_Uncert_grid, pi_S_Uncert]=discretizeAR1_FarmerToda(0, Params.rhoS2, Params.sigma_S2epsilon, n_S(2));
    S_Uncert_grid=exp(S_Uncert_grid);
    [meanS_Uncert,~,~,~]=MarkovChainMoments(S_Uncert_grid,pi_S_Uncert);
    S_Uncert_grid=S_Uncert_grid/meanS_Uncert; % make sure E[S2]=1
end

% (log) S_TFP is an AR(1) process, the innovations of which depend on S_Uncert
for iir=1:n_S(2) % reverse order
    ii=n_S(2)-iir+1; % Do in reverse order, because want S_TFP grid based on biggest uncertainty

    if n_S(1)==1 % just for debug/testing purposes
        S_TFP_grid=1;
        pi_S_TFP=1;
    elseif n_S(1)==2 % just for debug/testing purposes
        S_TFP_grid=[0.99; 1.01];
        pi_S_TFP=[0.9, 0.1; 0.1, 0.9];
    else
        if iir==1
            tauchenoptions=struct();
        end
        % Actual model
        [S_TFP_grid_raw, pi_S_TFP]=discretizeAR1_Tauchen(0, Params.rhoS1, gather(S_Uncert_grid(ii))*Params.sigma_S1epsilon, n_S(1),1,tauchenoptions);
        S_TFP_grid=exp(S_TFP_grid_raw); % S1, rather than log(S1)
        [meanS_TFP,~,~,~]=MarkovChainMoments(S_TFP_grid,pi_S_TFP);
        S_TFP_grid=S_TFP_grid/meanS_TFP; % make sure E[S1|S2]=1
        if iir==1
            tauchenoptions.z_grid=S_TFP_grid_raw; % Use the same grid on S_TFP for all the different S_Uncert, only the probabilities will change
        end
    end

    pi_S((1:1:n_S(1))+n_S(1)*(ii-1),:)=repmat(pi_S_TFP,1,n_S(2)).*repelem(pi_S_Uncert(1,:),1,n_S(1));
    S_gridvals((1:1:n_S(1))+n_S(1)*(ii-1),:)=[S_TFP_grid,S_Uncert_grid(ii)*ones(n_S(1),1)];
end

% (log) z is an AR(1) process, the innovations of which depend on S_Uncert
pi_z_S=zeros(prod(n_z),prod(n_z),n_S(2)); % repelem over n_S(1) later
z_grid_S=zeros(sum(n_z),n_S(2));
for iir=1:n_S(2) % reverse order
    ii=n_S(2)-iir+1; % Do in reverse order, because want S_TFP grid based on biggest uncertainty

    if iir==1
        tauchenoptions=struct();
    end
    % Actual model
    [z_grid_raw, pi_z]=discretizeAR1_Tauchen(0, Params.rho_z, gather(S_Uncert_grid(ii))*Params.sigma_zepsilon, n_z,1,tauchenoptions);
    z_grid=exp(z_grid_raw); % S1, rather than log(S1)
    [meanS_TFP,~,~,~]=MarkovChainMoments(z_grid,pi_z);
    z_grid=z_grid/meanS_TFP; % make sure E[S1|S2]=1
    if iir==1
        tauchenoptions.z_grid=z_grid_raw; % Use the same grid on z for all the different S_Uncert, only the probabilities will change
    end

    pi_z_S(:,:,ii)=pi_z;
    z_grid_S(:,ii)=z_grid;
end
pi_z_S=repelem(pi_z_S,1,1,n_S(1));
z_grid_S=repelem(z_grid_S,1,n_S(1));


%% Return Fn
DiscountFactorParamNames={'beta'};

if separate_rw==0
    ReturnFn=@(aprime,a,z,r,sigma,alpha,delta)...
        HARBCmodel2separaterw_ReturnFn(aprime,a,z,r,sigma,alpha,delta);
elseif separate_rw==1
    ReturnFn=@(aprime,a,z,r,w,sigma,fakel)...
        HARBCmodel2_ReturnFn(aprime,a,z,r,w,sigma,fakel);
end
% Note: It is possible to include S_TFP and S_Uncert as inputs to ReturnFn, just as if they were parameters

%%
FnsToEvaluate.K=@(aprime,a,z) a;
FnsToEvaluate.L=@(aprime,a,z) z;
% Note: It is possible to include S_TFP and S_Uncert as inputs to FnsToEvaluate, just as if they were parameters

%% General Eqm
if separate_rw==0
    GEPriceParamNames={'r'};
    GeneralEqmEqns.capitalmarket=@(r,S_TFP,K,L,alpha,delta) r-(alpha*S_TFP*(K^(alpha-1))*(L^(1-alpha))-delta);
elseif separate_rw==1
    GEPriceParamNames={'r','w'};
    GeneralEqmEqns.capitalmarket=@(r,S_TFP,K,L,alpha,delta) r-(alpha*S_TFP*(K^(alpha-1))*(L^(1-alpha))-delta);
    GeneralEqmEqns.labourmarket=@(w,S_TFP,K,L,alpha) w-((1-alpha)*S_TFP*(K^(alpha))*(L^(-alpha)));
end

%% Use divide-and-conquer together with grid interpolation layer when solving value fn problem
vfoptions.divideandconquer=1;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=50;
simoptions.gridinterplayer=vfoptions.gridinterplayer;
simoptions.ngridinterp=vfoptions.ngridinterp;


%% Test out the stationary general eqm (tends to be a good idea to do this to make sure we have a calibration that seems reasonable, before we turn to the aggregate shocks)
% It will just use pi_z from the 'first' uncertainty shock [I have run with
% pi_z from 'last' uncertainty shock, noone cleared 50 assets, let alone
% the top grid point of 100]

% vfoptions.divideandconquer=0;
% heteroagentoptions.verbose=1;
% heteroagentoptions.toleranceGEcondns=1e-6
% heteroagentoptions.intermediateEqns.Y=@(S_TFP,K,alpha,L) S_TFP*(K^alpha)*(L^(1-alpha));
% Params.S_TFP=1;
% Params.S_Uncert=1;
% [p_eqm,GEcondn]=HeteroAgentStationaryEqm_InfHorz(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% if separate_rw==0
%     Params.r=p_eqm.r;
% elseif separate_rw==1
%     Params.r=p_eqm.r;
%     Params.w=p_eqm.w;
% end
% 
% [V,Policy]=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
% StationaryDist=StationaryDist_InfHorz(Policy,n_d,n_a,n_z,pi_z,simoptions);
% cumDist=cumsum(sum(StationaryDist,2));
% figure(1);
% plot(a_grid,cumDist)



%% Solve the recursive general equilibrium with aggregate shocks
% T=1000; % number of time periods used in the matched-expectations path
recursiveeqmoptions.burnin=50; % this many periods are added to each end get ignored during the matching step

recursiveeqmoptions.verbose=2; % Give feedback
recursiveeqmoptions.heteroagentoptions.verbose=1; % Turn on verbose while solving for the initial guess

% We use a shooting-algorithm to update the general eqm prices, the following lines set this up
recursiveeqmoptions.GEnewprice=3;
if separate_rw==0
    recursiveeqmoptions.GEnewprice3.howtoupdate={... % same thing as when setting up a transition path, or using fminalgo=5 for the stationary general eqm
        'capitalmarket','r',0,0.1 ... % capitalmarket GE condition is positive when r is too big, so substract
        };
elseif separate_rw==1
    recursiveeqmoptions.GEnewprice3.howtoupdate={... % same thing as when setting up a transition path, or using fminalgo=5 for the stationary general eqm
        'labourmarket','w',0,0.05; ... % labourmarket GE condition is positive when w is too big, so substract
        'capitalmarket','r',0,0.1 ... % capitalmarket GE condition is positive when r is too big, so substract
        };
end


% Graphs at each iteration on the matched expectations path that show what is going on
recursiveeqmoptions.graphaggvarspath=1;
recursiveeqmoptions.graphpricepath=1;

% Solves using the matched-expectations path algorithm of Hanbaek Lee
time1=datetime;
tic;
GeneralizedTransitionFn=RecursiveGeneralEqmWithAggShocks_InfHorz(T,n_d,n_a,n_z,n_S,d_grid,a_grid,z_grid_S,S_gridvals,pi_z_S,pi_S,ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, GEPriceParamNames, AggShockNames, recursiveeqmoptions,vfoptions,simoptions);
GEtime=toc
time2=datetime;

% Started
time1
% Finished 
time2


