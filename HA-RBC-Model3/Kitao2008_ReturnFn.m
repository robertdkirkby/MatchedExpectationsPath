function F=Kitao2008_ReturnFn(aprime,eprime,a,e,eta,theta,S,sigma,d,upsilon1,upsilon2,r,w,delta,phi,tau_a0, tau_a1, tau_a2, tau_I, tau_c)

F=-Inf;

if e==0 % Workers' problem
    I=w*eta+r*a;
    
    TaxI=tau_a0*(I-(tau_a2+I^(-tau_a1))^(-1/tau_a1))+tau_I*I;

    c=(I+a-TaxI-aprime)/(1+tau_c);
    
    if c>0
        F=(c^(1-sigma))/(1-sigma); % CES utility fn
    end
    
    if aprime<0
        F=-Inf; % Borrowing constraint
    end
    
elseif e==1 % Entrepreneurs' problem

    % The entrepreneurs production problem is static.
    % See http://discourse.vfitoolkit.com/t/on-solving-models-with-entrepreneurs/296
    [k,n,output]=Kitao2008_StaticEntrepreneurProblem(a,theta,r,w,S,d,phi,delta,upsilon1,upsilon2);
    
    rbar=r;
    if k>a
        rbar=r+phi; % Cost of borrowing
    end
    
    I=output-delta*k-rbar*(k-a)-w*(n-eta)*(n>eta); % Note: (n-eta)*(n>eta) is just max(n-eta,0)
    % Note: the -rbar*(k-a) is because
    % "If the agent is a net borrower, i.e., k>a, the interest payment for
    % the borrowing is deducted as operational costs. If only part of his
    % assets are invested, i.e., k<=a, the remaining (a-k) earns a riskless
    % return which is added to the tax base of the entrepreneur as capital
    % income."
    
    TaxI=0;
    if I>0
        TaxI=tau_a0*(I-(tau_a2+I^(-tau_a1))^(-1/tau_a1))+tau_I*I;
    end

    profit=I+a-TaxI; % Eqn (7) of Kitao (2008)
    % Sub eqns (8) in (7) to get the above
    % Note: in eqn (7) the +(1-delta)k-(1+rbar)(k-a); we already have -delta*k-rbar(k-a) in I (eqn 8), so left with +k-(k-a) which becomes +a
    
    c=(profit-aprime)/(1+tau_c);
    
    if c>0
        F=(c^(1-sigma))/(1-sigma); % CES utility fn
    end
    
    if aprime<0
        F=-Inf; % Borrowing constraint
    end
        
end

end