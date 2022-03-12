function [wt] = My_Perceptron_L2(x,y)
%% Using Least squares to compute the decision boundery
% x -- data points; y -- label
% wt -- output parameter
% example: 
%          load('data_Perc.mat');
%          [wt] = My_Perceptron_L2(x,y);

Len = length(y);
Data = [ones(Len,1) x'];
wt = Data'*Data\Data'*y';
error = y - wt'*Data';
error = error.^2;
 
xp = x(:,y>0);
xn = x(:,y<0);
figure(1)
subplot(221)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid

axis([0 12 0 8])
axis square
xlabel('(a) Training Data')

hold off  
subplot(222)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid
p1 = 0;
p2 = (-wt(2)*p1-wt(1))/wt(3);
q1 = 12;
q2 = (-wt(2)*q1-wt(1))/wt(3);
plot([p1 q1],[p2 q2],'m--','linewidth',1.5)
axis([0 12 0 8])
axis square
xlabel('(b) Decision Boundary of Perceptron')

subplot(212)
stem(error,'b','linewidth',1.5);
xlabel('(c) Error of each points')
hold off  



