clear
clc
% read in the training data x and the labels y
load('data_Perc.mat');
figure(1)
hold on
Len = length(y);
% compute the label of each points in  axis([0,12,0,8]) by KNN
% draw each points: Red for y = -1 and Blue for y = 1
% draw the decision region of point (4,4,2) with color 'm'
for i = 0:0.1:12
    for j = 0:0.1:8
        temp = [i;j];
        temp2 = temp*ones(1,Len) - x;
        temp_sqr = sqrt(sum(temp2.^2,1));
        [~,inx] = sort(temp_sqr);
        if y(inx(1))==1
            plot(i,j,'b.','linewidth',2);
        else
            plot(i,j,'r.','linewidth',2);
            if inx(1)==6
                plot(i,j,'m.','linewidth',2);
            end
        end
    end
end
xp = x(:,y>0);
xn = x(:,y<0);
plot(xp(1,:),xp(2,:),'yo','linewidth',2)
plot(xn(1,:),xn(2,:),'gx','linewidth',2)

figure(2)
hold on
Len = length(y);
% compute the label of each points in  axis([0,12,0,8]) by KNN
% draw each points: Red for y = -1 and Blue for y = 1
% draw the decision region of point (4,4,2) with color 'm'
for i = 0:0.1:12
    for j = 0:0.1:8
        temp = [i;j];
        temp2 = temp*ones(1,Len) - x;
        temp_sqr = sqrt(sum(temp2.^2,1));
        [~,inx] = sort(temp_sqr);
        if y(inx(1))==1
            plot(i,j,'b.','linewidth',2);
        else
            plot(i,j,'r.','linewidth',2);
            if inx(1)==6
                plot(i,j,'m.','linewidth',2);
            end
        end
    end
end
xp = x(:,y>0);
xn = x(:,y<0);
plot(xp(1,:),xp(2,:),'yo','linewidth',2)
plot(xn(1,:),xn(2,:),'gx','linewidth',2)
