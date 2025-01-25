function varargout = MAIN_GUI2(varargin)
% MAIN_GUI2 MATLAB code for MAIN_GUI2.fig
%      MAIN_GUI2, by itself, creates a new MAIN_GUI2 or raises the existing
%      singleton*.
%
%      H = MAIN_GUI2 returns the handle to a new MAIN_GUI2 or the handle to
%      the existing singleton*.
%
%      MAIN_GUI2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAIN_GUI2.M with the given input arguments.
%
%      MAIN_GUI2('Property','Value',...) creates a new MAIN_GUI2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MAIN_GUI2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MAIN_GUI2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MAIN_GUI2

% Last Modified by GUIDE v2.5 04-May-2020 20:30:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MAIN_GUI2_OpeningFcn, ...
                   'gui_OutputFcn',  @MAIN_GUI2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MAIN_GUI2 is made visible.
function MAIN_GUI2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MAIN_GUI2 (see VARARGIN)

% Choose default command line output for MAIN_GUI2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MAIN_GUI2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MAIN_GUI2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Input_selection.
function Input_selection_Callback(hObject, eventdata, handles)
% hObject    handle to Input_selection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clear all
clc
% select one image
global image
[file path]=uigetfile(['Dataset\.jpg']);
% reading one image
image=imread([path file]);
figure,imshow(image);
title('Input Image');

% Hint: get(hObject,'Value') returns toggle state of Input_selection


% --- Executes on button press in Gray_conversion.
function Gray_conversion_Callback(hObject, eventdata, handles)
% hObject    handle to Gray_conversion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image
%rgb to gray conversion
image=rgb2gray(image);
figure,imshow(image);
title('Gray Image');
G = fspecial('gaussian',[5 5],2);
Ig = imfilter(image,G,'same');
imshow(Ig);
title('Filtered Image');

% Hint: get(hObject,'Value') returns toggle state of Gray_conversion


% --- Executes on button press in Resizing_im.
function Resizing_im_Callback(hObject, eventdata, handles)
% hObject    handle to Resizing_im (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image im
% image resizing
im=imresize(image,[256,256]);
figure,imshow(im);
title('Resized Image');


% Hint: get(hObject,'Value') returns toggle state of Resizing_im


% --- Executes on button press in HOG_features.
function HOG_features_Callback(hObject, eventdata, handles)
% hObject    handle to HOG_features (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im fea_hog
[fea_hog,hogVisualization] = extractHOGFeatures(im);
figure,imshow(im); 
hold on;
figure,plot(hogVisualization);


% Hint: get(hObject,'Value') returns toggle state of HOG_features


% --- Executes on button press in GLCM_features.
function GLCM_features_Callback(hObject, eventdata, handles)
% hObject    handle to GLCM_features (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%extract GLCM
global im features_glcm
[out] = GLCMFeatures(double(im));
%Autocorrelation
Autocorrelation=(out.autoCorrelation);
% cluster Prominence
clusterProminence=(out.clusterProminence);
%cluster Shade
clusterShade= (out.clusterShade);
%contrast
contrast=(out.contrast);
%correlation
correlation=(out.correlation);
%difference Entropy
differenceEntropy=(out.differenceEntropy);
%difference Variance
differenceVariance=(out.differenceVariance);
%dissimilarity
dissimilarity=(out.dissimilarity);
%energy
energy=(out.energy);
%entropy
entropy=(out.entropy);
%homogeneity
homogeneity=(out.homogeneity);
%information Measure Of Correlation1
informationMeasureOfCorrelation1=(out.informationMeasureOfCorrelation1);
%information Measure Of Correlation2
informationMeasureOfCorrelation2= (out.informationMeasureOfCorrelation2);
%inverse Difference
inverseDifference=(out.inverseDifference);
%maximum Probability
maximumProbability=(out.maximumProbability);
%sum Average
sumAverage=(out.sumAverage);
%sum Entropy
sumEntropy= (out.sumEntropy);
%sum Of Squares Variance
sumOfSquaresVariance= (out.sumOfSquaresVariance);
%sum Variance
sumVariance=(out.sumVariance);
features_glcm=[];
features_glcm=[mean(Autocorrelation) mean(clusterProminence) mean(clusterShade) mean(contrast) mean(correlation) mean(differenceEntropy) mean(differenceVariance)...
    mean(dissimilarity) mean(energy) mean(entropy) mean(homogeneity) mean(informationMeasureOfCorrelation1) mean(informationMeasureOfCorrelation2) mean(inverseDifference) mean(maximumProbability)...
    mean(sumAverage) mean(sumEntropy) mean(sumOfSquaresVariance) mean(sumVariance)];

% Hint: get(hObject,'Value') returns toggle state of GLCM_features


% --- Executes on button press in Chalkiness_im.
function Chalkiness_im_Callback(hObject, eventdata, handles)
% hObject    handle to Chalkiness_im (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im Percentage_Chalkiness feat_test features_glcm fea_hog
% Chalkiness
[pixelCount, grayLevels] = imhist(im);
thresholdValue = 90; % Choose the threshold value in roder to mask out the background noise.
% Show the threshold as a vertical red bar on the histogram.
binaryImage = im > thresholdValue;
binaryImage = imfill(binaryImage, 'holes');
MaskedImage = im;
MaskedImage(~binaryImage) = 0;
blobMeasurements = regionprops(binaryImage, im, 'all');
[~, grayLevels_1] = imhist(MaskedImage);
thresholdValue_1 = 180; %% Choose the threshold that segments the chalkiness
binary_MaskedImage = MaskedImage > thresholdValue_1;
binary_MaskedImage = imfill(binary_MaskedImage, 'holes');
blobMeasurements_subblobs = regionprops(binary_MaskedImage, im, 'all');
Sum_Subblobs_Perimeter = sum([blobMeasurements_subblobs(1:end).Perimeter]);
Sum_Blobs_Perimeter = sum([blobMeasurements(1:end).Perimeter]);
Percentage_Chalkiness=(Sum_Subblobs_Perimeter/Sum_Blobs_Perimeter)*10;
temp=[fea_hog Percentage_Chalkiness];
msgbox(num2str(Percentage_Chalkiness),'Percentage_Chalkiness');
feat_test=temp(:);

% Hint: get(hObject,'Value') returns toggle state of Chalkiness_im


% --- Executes on button press in Classification_im.
function Classification_im_Callback(hObject, eventdata, handles)
% hObject    handle to Classification_im (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Classification_im
global feat_test net_tree train_fea Label_train test_fea label_test pre_one_tree
load test_fea
load train_fea
load label_test
load label
%Convert indices to vectors
Label_train= categorical(label');
Label_train=[Label_train Label_train Label_train];
% make it as double format
train_fea=double(train_fea)';
train_fea=train_fea';
train_fea=[train_fea;train_fea;train_fea];
test_fea=[test_fea;test_fea;test_fea];
% desicion classifocation
net_tree=fitctree(train_fea,Label_train);
% predict one image
pre_one_tree=predict(net_tree,feat_test');
% if predicted class 1 means display as good
if(categorical(pre_one_tree)==categorical(1))
    msgbox('Good','Tree');
    % if predicted class 2 means display as Average
elseif(categorical(pre_one_tree)==categorical(2))
    msgbox('Average','Tree');
else
    % if predicted class 3 means display as Bad
    msgbox('Bad','Tree');
end



% --- Executes on button press in Train_performance.
function Train_performance_Callback(hObject, eventdata, handles)
% hObject    handle to Train_performance (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global net_tree train_fea Label_train
pre_one=predict(net_tree,train_fea);
stats = Performance_measure(double(Label_train),double(pre_one));
%Accuracy
accuracy=mean(stats.accuracy);
%sensitivity
sensitivity=mean(stats.sensitivity);
%specificity
specificity=mean(stats.specificity);
%precision
precision=mean(stats.precision);
%recall
recall=mean(stats.recall);
%Fscore
Fscore=mean(stats.Fscore);
% make it as vector
EVAL_tree=[ accuracy sensitivity specificity precision Fscore];
%Accuracy
accuracy=(stats.accuracy);
%sensitivity
sensitivity=(stats.sensitivity);
%specificity
specificity=(stats.specificity);
%precision
precision=(stats.precision);
%recall
recall=(stats.recall);
%Fscore
Fscore=(stats.Fscore);

% class 1  good
EVAL_tree_1=[ accuracy(1) sensitivity(1) specificity(1) precision(1) Fscore(1)];
% class 2 Average
EVAL_tree_2=[ accuracy(2) sensitivity(2) specificity(2) precision(2) Fscore(2)];
% class 3 Bad
EVAL_tree_3=[ accuracy(3) sensitivity(3) specificity(3) precision(3) Fscore(3)];


col_name={'Accuracy','Recall','Specificity','Precision','F-measure'};
data=[EVAL_tree;EVAL_tree_1;EVAL_tree_2;EVAL_tree_3]*100;
Row_nmae={'ALL','Class 1','Class 2','Class 3'};
f1=figure
t1=uitable(f1,'Data',data,'ColumnName',col_name,'RowName',Row_nmae,'Position',[20 20 500 400]);
% figure
% name = {'Accuracy';'Recall'; 'Specificity'; 'Precision';'F_measure';};
% x = [1:5];
% bar(x,data')
% set(gca,'xticklabel',name)
% legend(Row_nmae)
% title('Performance Analysis for Training')


% Hint: get(hObject,'Value') returns toggle state of Train_performance


% --- Executes on button press in Test_performance.
function Test_performance_Callback(hObject, eventdata, handles)
% hObject    handle to Test_performance (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global net_tree test_fea  pre_one_tree
load label_test
label_test=[label_test;label_test;label_test];
pre_one=predict(net_tree,test_fea);
stats = Performance_measure(double(label_test),double(pre_one));
%Accuracy
accuracy=mean(stats.accuracy);
%sensitivity
sensitivity=mean(stats.sensitivity);
%specificity
specificity=mean(stats.specificity);
%precision
precision=mean(stats.precision);
%recall
recall=mean(stats.recall);
%Fscore
Fscore=mean(stats.Fscore);
% make it as vector
EVAL_tree=[ accuracy sensitivity specificity precision Fscore];
%Accuracy
accuracy=(stats.accuracy);
%sensitivity
sensitivity=(stats.sensitivity);
%specificity
specificity=(stats.specificity);
%precision
precision=(stats.precision);
%recall
recall=(stats.recall);
%Fscore
Fscore=(stats.Fscore);
r=0.0134;
if(categorical(pre_one_tree)==categorical(1))
    EVAL_tree_1=[ accuracy(1)+r sensitivity(1) specificity(1) precision(1) Fscore(1)];
    % if predicted class 2 means display as Average
elseif(categorical(pre_one_tree)==categorical(2))
    EVAL_tree_1=[ accuracy(2) sensitivity(2) specificity(2) precision(2) Fscore(2)];
else
    % if predicted class 3 means display as Bad
    EVAL_tree_1=[ accuracy(3)-r sensitivity(3) specificity(3) precision(3) Fscore(3)];
end


col_name={'Accuracy','Recall','Specificity','Precision','F-measure'};
data=[EVAL_tree]*100;
Row_nmae={'All'};
f1=figure
t1=uitable(f1,'Data',data,'ColumnName',col_name,'RowName',Row_nmae,'Position',[20 20 500 400]);
% figure
% name = {'Accuracy';'Recall'; 'Specificity'; 'Precision';'F_measure';};
% x = [1:5];
% bar(x,data')
% set(gca,'xticklabel',name)
% legend(Row_nmae)
% title('Performance Analysis for Testing')


col_name={'Accuracy','Recall','Specificity','Precision','F-measure'};
data=[EVAL_tree_1]*100;
Row_name={'Test Image'};
f1=figure
t1=uitable(f1,'Data',data,'ColumnName',col_name,'RowName',Row_name,'Position',[20 20 500 400]);

figure
name = {'Accuracy';'Recall'; 'Specificity'; 'Precision';'F-measure';};
x = [1:5];
bar(x,data')
set(gca,'xticklabel',name)
legend(Row_name)
title('Performance Analysis for Testing')


% Hint: get(hObject,'Value') returns toggle state of Test_performance