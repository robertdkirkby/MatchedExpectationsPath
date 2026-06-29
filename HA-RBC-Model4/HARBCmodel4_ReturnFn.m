function F=HARBCmodel4_ReturnFn(aprime,a,z,r,sigma,alpha,delta)

F=-Inf;

w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));

c=(1+r)*a+w*z-aprime; % Note: w*z + b*(1-z) with z=0 and 1 can be simplified to just w*z with z=b/w and z=1

if c>0
    if sigma==1
        F=log(c);
    else
        F=(c^(1-sigma) -1)/(1-sigma);
    end
end



end
