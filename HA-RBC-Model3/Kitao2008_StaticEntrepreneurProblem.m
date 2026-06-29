function [k,n,output]=Kitao2008_StaticEntrepreneurProblem(a,theta,r,w,S,d,phi,delta,upsilon1,upsilon2)
% The entrepreneurs production problem is static.
% See http://discourse.vfitoolkit.com/t/on-solving-models-with-entrepreneurs/296
% Note that we need to handle that there is a different r for borrowing and saving.
onegg=1-upsilon1-upsilon2; % just to simplify the below forumlaes

% Note: relative to Kitao (2008), just put S*theta in place of theta
% [Aggregate productivity multiplies entrepreneur productivity]

% Find capital used by entrepreneur if they borrow (so based on r+phi)
k_unc=(upsilon1/(r+phi+delta))^((1-upsilon1)/onegg) * (upsilon2/w)^(upsilon2/onegg) * (S*theta)^(1/onegg);
if k_unc>(1+d)*a % Collateral constraint
    k=(1+d)*a;
else
    k=k_unc;
end
% If k<a, switch to using the r for savings
if k<a
    k_r=(upsilon1/(r+delta))^((1-upsilon1)/onegg) * (upsilon2/w)^(upsilon2/onegg) * (S*theta)^(1/onegg); % This is same formula as k_unc, but based on r rather than r+phi
    if k_r>a
        k=a;
    else
        k=k_r;
    end
end

% Now find labor
n=((upsilon2*(S*theta)*k^upsilon1)/w)^(1/(1-upsilon2));

% Now that we have the production decisions for k and n, rest is straightforward
output=(S*theta)*(k^upsilon1)*(n^upsilon2);

end