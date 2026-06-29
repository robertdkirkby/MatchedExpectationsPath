function TaxRevenue=Kitao2008_TaxFn(aprime,eprime,a,e,eta,theta,S,d,upsilon1,upsilon2,r,w,delta,phi,tau_a0, tau_a1, tau_a2, tau_I, tau_c)
% Revenue of: Income tax plus consumption tax

TaxRevenue=0;
if e==0 % Workers' problem
    I=w*eta+r*a;
    
    TaxI=tau_a0*(I-(tau_a2+I^(-tau_a1))^(-1/tau_a1))+tau_I*I;

    c=(w*eta+(1+r)*a-TaxI-aprime)/(1+tau_c);
    
    TaxRevenue=tau_c*c+TaxI;
    
elseif e==1 % Entrepreneurs' problem

    [k,n,output]=Kitao2008_StaticEntrepreneurProblem(a,theta,r,w,S,d,phi,delta,upsilon1,upsilon2);
    
    rbar=r;
    if k>a
        rbar=r+phi;
    end
    
    I=output-delta*k-rbar*(k-a)-w*(n-eta)*(n>eta); % Note: (n-eta)*(n>eta) is just max(n-eta,0)
        
    TaxI=0;
    if I>0
        TaxI=tau_a0*(I-(tau_a2+I^(-tau_a1))^(-1/tau_a1))+tau_I*I;
    end

    profit=I+a-TaxI; % Eqn 5 of Kitao (2008)
    
    c=(profit-aprime)/(1+tau_c);
    
    TaxRevenue=tau_c*c+TaxI;
    
end



end