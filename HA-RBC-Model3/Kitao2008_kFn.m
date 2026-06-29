function k=Kitao2008_kFn(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi)
% Capital used by entrepreneur. Returns 0 if worker.

k=0; % just to make GPU happy

if e==0 % Workers' problem
    k=0;
elseif e==1 % Entrepreneurs' problem
    [k,~,~]=Kitao2008_StaticEntrepreneurProblem(a,theta,r,w,S,d,phi,delta,upsilon1,upsilon2);
end



end