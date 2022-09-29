function s = my_sig(W,x)
% sigmoid function

s = 1.0/(1+exp(-W'*x));