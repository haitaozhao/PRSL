clear;
clc;
t = cputime;

u_train = load('u1.base');

R_train = sparse(943,1682);
R_train(sub2ind(size(R_train),u_train(:,1),u_train(:,2))) = u_train(:,3);
RR_train = full(R_train);

[U,s,V] = svd(RR_train,'econ');

u1 = load('u1.test');
dim = 6;

Z = s(1:6,1:6) * V(:,1:6)';

u_test = zeros(20000,3);

for i = 1:20000
    user = u1(i,1);
    item = u1(i,2);
    [~,b] = find(RR_train(user,:)~=0);
    Temp = Z(:,b);
    for j = 1 : length(b)
        Temp(:,j) = Temp(:,j)/norm(Temp(:,j));
    end
    c = Z(:,item)'/norm(Z(:,item))*Temp;
    [~,d] = sort(c,'descend');
    u_test(i,1) = user;
    u_test(i,2) = item;
    u_test(i,3) = mean(RR_train(user,b(d(1:4))));
end

RMSE = 1/sqrt(20000)* norm(u1(:,3)-u_test(:,3))







