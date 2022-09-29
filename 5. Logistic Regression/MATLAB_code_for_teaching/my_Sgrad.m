function g = my_Sgrad(W,x,y)
%% Compute the gradient of NLL

g = (my_sig(W,x)-y)*x;

