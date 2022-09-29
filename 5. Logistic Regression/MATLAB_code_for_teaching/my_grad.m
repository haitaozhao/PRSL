function g = my_grad(W,X,Y)
%% Compute the gradient of NLL
len = length(Y);
sum = 0;

for i = 1 : len
    sum = sum + (my_sig(W,X(:,i))-Y(i))*X(:,i);
end

g =  1/len*sum;