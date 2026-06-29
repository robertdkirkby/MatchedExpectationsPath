function n=Kitao2008_nFn(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi)
% Capital used by entrepreneur. Returns 0 if worker.

n=0; % just to make GPU happy

if e==0 % Workers' problem
    n=0;
elseif e==1 % Entrepreneurs' problem
    [~,n,~]=Kitao2008_StaticEntrepreneurProblem(a,theta,r,w,S,d,phi,delta,upsilon1,upsilon2);
end



end