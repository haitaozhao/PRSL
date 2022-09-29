function [wt,t] = My_Perceptron(x,y,r,w0)
%% Batch algorithm of Perceptron
% x -- data points; y -- label
% r -- learning rate; w0 -- start point of the parameters
% wt -- output parameter; t -- iteration number
% example: 
%          load('data_Perc.mat');
%          [wt,t] = My_Perceptron(x,y,0.01,[0,0,0]');
t = 0;
%w0 = [0,0,0]';
Len = length(y);
Data = [ones(Len,1) x'];
Temp = Data*w0;
bTemp = Temp>=0;
by = y==1;
inx = by ~= bTemp';
error = [];
while sum(inx)> 0 & t<10000
    deltaW =  Data(inx,:)'* y(inx)'; 
    wt = w0 + r* deltaW;
    t = t + 1;
    error(t) = - (Data(inx,:)*w0/norm(w0))'* y(inx)';
    w0 = wt;
    Temp = Data*w0;
    bTemp = [];
    by = [];
    bTemp = Temp>=0;
    by = y==1;
    inx = by ~= bTemp';
end
%% plot the decision funciton 
xp = x(:,y>0);
xn = x(:,y<0);
figure(1)
subplot(221)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid on

axis([0 12 0 8])
axis square
xlabel('(a) Training Data')

hold off  
subplot(222)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid on
p1 = 0;
p2 = (-wt(2)*p1-wt(1))/wt(3);
q1 = 12;
q2 = (-wt(2)*q1-wt(1))/wt(3);
plot([p1 q1],[p2 q2],'k-','linewidth',1.5)
axis([0 12 0 8])
axis square
xlabel('(b) Decision Boundary of Perceptron')

subplot(212)
tt = 1:10:t;
plot(tt,error(tt),'b-','linewidth',1.5);
xlabel('(c) Objective function')
grid on
hold off  



