function f = my_fun(W,X,Y)
%% Compute NLL
len = length(Y);
sum = 0;

for i = 1 : len
    sum = sum + Y(i)*log(my_sig(W,X(:,i)))+(1-Y(i))*log(1-my_sig(W,X(:,i)));
end

f =  - 1/len*sum;