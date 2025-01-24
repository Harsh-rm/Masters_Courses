function stats = Performance_measure(group,grouphat)
% INPUT
% group = true class labels
% grouphat = predicted class labels
%
% OR INPUT
% stats = confusionmatStats(group);
% group = confusion matrix from matlab function (confusionmat)
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____| 
%       Actual  p |     |     |
%      Classes  n |     |     |
%
% stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
% stats.precision = TP / (TP + FP)                  % for each class label
% stats.sensitivity = TP / (TP + FN)                % for each class label
% stats.specificity = TN / (FP + TN)                % for each class label
% stats.recall = sensitivity                        % for each class label
% stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 
field1 = 'confusionMat';
% determing confudion matrix
if nargin < 2
    value1 = group;
else
    [value1,gorder] = confusionmat(group,grouphat);
end
% calculating total number of classes
numOfClasses = size(value1,1);
% calculating total number of samples
totalSamples = sum(sum(value1));
% initially assign zero for all parameter
[TP,TN,FP,FN,accuracy,sensitivity,specificity,precision,f_score] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   % extracting true positive from confusion matrix for all class
   TP(class) = value1(class,class);
   tempMat = value1;
   % remove column
   tempMat(:,class) = []; 
   % remove row
   tempMat(class,:) = []; 
   % extracting true negative from confusion matrix for all class
   TN(class) = sum(sum(tempMat));
   % extracting False positive from confusion matrix for all class
   FP(class) = sum(value1(:,class))-TP(class);
   % extracting False negative from confusion matrix for all class
   FN(class) = sum(value1(class,:))-TP(class);
end
for class = 1:numOfClasses
    % calculating Accuracy
    accuracy(class) = (TP(class) + TN(class)) / totalSamples;
    % calculating sensitivity
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    % calculating specificity
    specificity(class) = TN(class) / (FP(class) + TN(class));
    % calculating precision
    precision(class) = TP(class) / (TP(class) + FP(class));
    % calculating f_score
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
end
field2 = 'accuracy';  value2 = accuracy;
field3 = 'sensitivity';  value3 = sensitivity;
field4 = 'specificity';  value4 = specificity;
field5 = 'precision';  value5 = precision;
field6 = 'recall';  value6 = sensitivity;
field7 = 'Fscore';  value7 = f_score;
stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7);
if exist('gorder','var')
    stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7,'groupOrder',gorder);
end