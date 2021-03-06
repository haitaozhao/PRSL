clear;
clc;
t = cputime;

u_train = load('u1.base');

R_train = sparse(943,1682);
RT_train = sparse(943,1682);
R_train(sub2ind(size(R_train),u_train(:,1),u_train(:,2))) = u_train(:,3);
Tempt = u_train(:,4);
[Tt,m,s] = zscore(Tempt);
Tt = (Tt - min(Tt))/(max(Tt)-min(Tt));
RT_train(sub2ind(size(R_train),u_train(:,1),u_train(:,2))) = Tt;

RR_train = full(R_train);

[U,s,V] = svd(RR_train,'econ');

u1 = load('u1.test');
dim = 80;

Z = sqrt(s(1:dim,1:dim)) * V(:,1:dim)';
ZZ = U(:,1:dim) * (s(1:dim,1:dim));

u_test = zeros(20000,3);

for i = 1:20000
    user = u1(i,1);
    item = u1(i,2);
    [~,b] = find(RR_train(user,:)~=0);
    [bb,~] = find(RR_train(:,item)~=0);
    Temp = Z(:,b);
    TempU = ZZ(bb,:);
    for j = 1 : length(b)
        Temp(:,j) = Temp(:,j)/norm(Temp(:,j));
    end
    
    for j = 1 : length(bb)
        TempU(j,:) = TempU(j,:)/norm(TempU(j,:));
    end
    c = Z(:,item)'/norm(Z(:,item))*Temp;
    cc = ZZ(user,:)/norm(ZZ(user,:))*TempU';
    [~,d] = sort(c,'descend');
    [~,dd] = sort(cc,'descend');
    u_test(i,1) = user;
    u_test(i,2) = item;
    if length(d)>20
        len = 20;
    else
        len = length(d);
    end
    Sim = c(d(1:len));
    R_temp = RR_train(user,b(d(1:len)));
    RT_temp = RT_train(user,b(d(1:len)))+10;
    Sim = Sim.*RT_temp;
    
    if length(dd)>10
        lenU = 10;
    else
        lenU = length(dd);
    end
    SimU = cc(dd(1:lenU));
    R_tempU = RR_train(bb(dd(1:lenU)),item);
    u_test(i,3) = 0.6* R_temp*Sim'/sum(Sim) + 0.4 * R_tempU'*SimU'/sum(SimU) ;

    
    if isnan(u_test(i,3))
        u_test(i,3) = 3;
    end
end

RMSE = 1/sqrt(20000)* norm(u1(:,3)-u_test(:,3))




