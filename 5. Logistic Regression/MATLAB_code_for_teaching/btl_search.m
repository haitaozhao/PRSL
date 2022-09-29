function alphak = btl_search(f,g,W,X,Y,rho,gamma)
% Backtracking line search
% See Algorithm 3.1 on page 37 of Nocedal and Wright

% Input Parameters :
% f: MATLAB file that returns function value
% g: The gradient of f
% x: previous iterate
% rho : parameter between 0 and 0.5, usually rho = 0.1
% gamma: parameter between 0 and 1 , usually 0.5
% Output :
% alphak: step length calculated by algorithm

alphak = 1;
fk = f(W,X,Y);
gk = g(W,X,Y);
d = -gk;
W_new = W + alphak*d;
fk1 = f(W_new,X,Y);
while fk1 > fk + rho*alphak*(gk'*d)
  alphak = alphak*gamma;
  W_new = W + alphak*d;
  fk1 = f(W_new,X,Y);
end