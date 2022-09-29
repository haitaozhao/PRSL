%% read in the data
Train = readtable('train.csv','Format','%f%f%f%q%s%f%f%f%q%f%q%s');
Test = readtable('test.csv','Format','%f%f%q%s%f%f%f%q%f%q%s');
disp(Train(1:5,[2:3 5:8 10:11]))

%% Sex information is quite important, we can use it to desigh the classifier
disp(grpstats(Train(:,{'Survived','Sex'}), 'Sex'))

gendermdl = grpstats(Train(:,{'Survived','Sex'}), {'Survived','Sex'});
all_female = (gendermdl.GroupCount('0_male') + gendermdl.GroupCount('1_female'))...
    / sum(gendermdl.GroupCount)

%% Data preprocessing. Some values are not available and others are not  reasonable
Train.Fare(Train.Fare == 0) = NaN;      % treat 0 fare as NaN
Test.Fare(Test.Fare == 0) = NaN;        % treat 0 fare as NaN
vars = Train.Properties.VariableNames;  % extract column names
figure
colormap(summer)
imagesc(ismissing(Train))
set(gca,'XTick', 1:12,'XTickLabel',{'ID','Survived','Pc','Name','Sex','Age','Sib','Par','Tic','Fare','Cab','Emb'});

%% fill in the missing ages with the average
avgAge = nanmean(Train.Age)             % get average age
Train.Age(isnan(Train.Age)) = avgAge;   % replace NaN with the average
Test.Age(isnan(Test.Age)) = avgAge; 

%% fill in the missing fares according to their Pclass and then fill in the average of that Pclass
fare = grpstats(Train(:,{'Pclass','Fare'}),'Pclass');   % get class average
disp(fare)
for i = 1:height(fare) % for each |Pclass|
    % apply the class average to missing values
    Train.Fare(Train.Pclass == i & isnan(Train.Fare)) = fare.mean_Fare(i);
    Test.Fare(Test.Pclass == i & isnan(Test.Fare)) = fare.mean_Fare(i);
end


%% about cabin
% tokenize the text string by white space
train_cabins = cellfun(@strsplit, Train.Cabin, 'UniformOutput', false);
test_cabins = cellfun(@strsplit, Test.Cabin, 'UniformOutput', false);

% count the number of tokens
Train.nCabins = cellfun(@length, train_cabins);
Test.nCabins = cellfun(@length, test_cabins);

% deal with exceptions - only the first class people had multiple cabins
Train.nCabins(Train.Pclass ~= 1 & Train.nCabins > 1,:) = 1;
Test.nCabins(Test.Pclass ~= 1 & Test.nCabins > 1,:) = 1;

% if |Cabin| is empty, then |nCabins| should be 0
Train.nCabins(cellfun(@isempty, Train.Cabin)) = 0;
Test.nCabins(cellfun(@isempty, Test.Cabin)) = 0;

%% fill in the Embarked with the most fequent Embarked place
% get most frequent value
disp(grpstats(Train(:,{'Survived','Embarked'}), 'Embarked'))

% apply it to missling value
for i = 1 : 891
	if isempty(Train.Embarked{i})
		Train.Embarked{i}='S';
	end
end

for i = 1 : 418
	if isempty(Test.Embarked{i})
		Test.Embarked{i}='S';
	end
end

% convert the data type from categorical to double
Train.Embarked = double(cell2mat(Train.Embarked));
Test.Embarked = double(cell2mat(Test.Embarked));

%% type of Set changed from logic to double
for i = 1 : 891
	if strcmp(Train.Sex{i} ,'male')
		Train.Sex{i}=1;
	else
		Train.Sex{i}=0;
	end
end

for i = 1 : 418
	if strcmp(Test.Sex{i} ,'male')
		Test.Sex{i}=1;
	else
		Test.Sex{i}=0;
	end
end
Train.Sex = cell2mat(Train.Sex);
Test.Sex = cell2mat(Test.Sex);

%% del some non useful features

Train(:,{'Name','Ticket','Cabin'}) = [];
Test(:,{'Name','Ticket','Cabin'}) = [];

%% feature analysis
 figure
 subplot(2,1,1)
 colormap(summer)
hist (Train.Age(Train.Survived == 0))   % age histogram of non-survivers
h = findobj(gca,'Type','patch');
set(h,'FaceColor',[1 .5 .5])
legend('Not Survived');

subplot(2,1,2)
hist (Train.Age(Train.Survived == 1))   % age histogram of survivers
legend('Survived');

%% random forest
Y_train = Train.Survived;                   % slice response variable
X_train = Train(:,3:end);                   % select predictor variables
vars = X_train.Properties.VariableNames;    % get variable names
X_train = table2array(X_train);             % convert to a numeric matrix
X_test = table2array(Test(:,2:end));        % convert to a numeric matrix
categoricalPredictors = {'Pclass', 'Sex', 'Embarked'};
rng(1);                                     % for reproducibility
c = cvpartition(Y_train,'holdout', 0.30);   % 30%-holdout cross validation
RF = TreeBagger(200, X_train(training(c),:), Y_train(training(c)),...
    'PredictorNames', vars, 'Method','classification',...
    'CategoricalPredictors', categoricalPredictors, 'oobvarimp', 'on');

% compute the out-of-bag accuracy
oobAccuracy = 1 - oobError(RF, 'mode', 'ensemble')


[~,order] = sort(RF.OOBPermutedVarDeltaError);  % sort the metrics
figure
colormap(summer)
barh(RF.OOBPermutedVarDeltaError(order))
set(gca,'YTickLabel',vars(order))

%% Performance on Validation set

 [Yfit, Yscore] = predict(RF, X_train(test(c),:));       % use holdout data
cfm = confusionmat(Y_train(test(c)), str2double(Yfit)); % confusion matrix
cvAccuracy = sum(cfm(logical(eye(2))))/length(Yfit) 

% ROC curve
 posClass = strcmp(RF.ClassNames,'1');   % get the index of the positive class
curves = zeros(2,1); labels = cell(2,1);% pre-allocated variables
[rocX, rocY, ~, auc] = perfcurve(Y_train(test(c)),Yscore(:,posClass),'1');
figure
curves(1) = plot(rocX, rocY);           % use the perfcurve output to plot
labels{1} = sprintf('Random Forest - AUC: %.1f%%', auc*100);
curves(end) = refline(1,0); set(curves(end),'Color','r');
labels{end} = 'Reference Line - A random classifier';
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Plot')
legend(curves, labels, 'Location', 'SouthEast') 

%% Final submission
PassengerId = Test.PassengerId;             % extract Passenger Ids
Survived = predict(RF, X_test);             % generate response variable
Survived = str2double(Survived);            % convert to double
submission = table(PassengerId,Survived);   % combine them into a table
disp(submission(1:5,:))                     % preview the table
 writetable(submission,'e:\submission.csv')


