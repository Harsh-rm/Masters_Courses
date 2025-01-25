clc
close all
clear all
% % training image
for ii=1:1500
    ii
    %     [file path]=uigetfile(['Dataset\All\.jpg']);
    image=imread(['Dataset\Train1\(',num2str(ii),').jpg']);
    image=imresize(image,[256,256]);
    im=rgb2gray(image);
    % features extraction
    % extract hog featrures
    fea_hog=extractHOGFeatures(im);
%     %extract GLCM
    [out] = GLCMFeatures(double(im));
    Autocorrelation=(out.autoCorrelation);
    clusterProminence=(out.clusterProminence);
    clusterShade= (out.clusterShade);
    contrast=(out.contrast);
    correlation=(out.correlation);
    differenceEntropy=(out.differenceEntropy);
    differenceVariance=(out.differenceVariance);
    dissimilarity=(out.dissimilarity);
    energy=(out.energy);
    entropy=(out.entropy);
    homogeneity=(out.homogeneity);
    informationMeasureOfCorrelation1=(out.informationMeasureOfCorrelation1);
    informationMeasureOfCorrelation2= (out.informationMeasureOfCorrelation2);
    inverseDifference=(out.inverseDifference);
    maximumProbability=(out.maximumProbability);
    sumAverage=(out.sumAverage);
    sumEntropy= (out.sumEntropy);
    sumOfSquaresVariance= (out.sumOfSquaresVariance);
    sumVariance=(out.sumVariance);
    features_glcm=[];
    features_glcm=[mean(Autocorrelation) mean(clusterProminence) mean(clusterShade) mean(contrast) mean(correlation) mean(differenceEntropy) mean(differenceVariance)...
        mean(dissimilarity) mean(energy) mean(entropy) mean(homogeneity) mean(informationMeasureOfCorrelation1) mean(informationMeasureOfCorrelation2) mean(inverseDifference) mean(maximumProbability)...
        mean(sumAverage) mean(sumEntropy) mean(sumOfSquaresVariance) mean(sumVariance)];
    % Chalkiness
    [pixelCount, grayLevels] = imhist(im);
    thresholdValue = 90; % Choose the threshold value in roder to mask out the background noise.
    % Show the threshold as a vertical red bar on the histogram.
    binaryImage = im > thresholdValue;
    binaryImage = imfill(binaryImage, 'holes');
    MaskedImage = im;
    MaskedImage(~binaryImage) = 0;
    blobMeasurements = regionprops(binaryImage, im, 'all');
    [pixelCount_1, grayLevels_1] = imhist(MaskedImage);
    thresholdValue_1 = 180; %% Choose the threshold that segments the chalkiness
    binary_MaskedImage = MaskedImage > thresholdValue_1;
    binary_MaskedImage = imfill(binary_MaskedImage, 'holes');
    blobMeasurements_subblobs = regionprops(binary_MaskedImage, im, 'all');
    Sum_Subblobs_Perimeter = sum([blobMeasurements_subblobs(1:end).Perimeter]);
    Sum_Blobs_Perimeter = sum([blobMeasurements(1:end).Perimeter]);
    Percentage_Chalkiness=(Sum_Subblobs_Perimeter/Sum_Blobs_Perimeter)*100;
    
    temp=[];
    temp=[fea_hog Percentage_Chalkiness];
    train_fea(ii,:)=temp;
    save train_fea train_fea
end
% tesung images
for ii=1:316
    ii
    %     [file path]=uigetfile(['Dataset\All\.jpg']);
    image=imread(['Dataset\Test1\(',num2str(ii),').jpg']);
    image=imresize(image,[256,256]);
    im=rgb2gray(image);
    % features extraction
    % extract hog featrures
    fea_hog=extractHOGFeatures(im);
    %extract GLCM
    [out] = GLCMFeatures(double(im));
    Autocorrelation=(out.autoCorrelation);
    clusterProminence=(out.clusterProminence);
    clusterShade= (out.clusterShade);
    contrast=(out.contrast);
    correlation=(out.correlation);
    differenceEntropy=(out.differenceEntropy);
    differenceVariance=(out.differenceVariance);
    dissimilarity=(out.dissimilarity);
    energy=(out.energy);
    entropy=(out.entropy);
    homogeneity=(out.homogeneity);
    informationMeasureOfCorrelation1=(out.informationMeasureOfCorrelation1);
    informationMeasureOfCorrelation2= (out.informationMeasureOfCorrelation2);
    inverseDifference=(out.inverseDifference);
    maximumProbability=(out.maximumProbability);
    sumAverage=(out.sumAverage);
    sumEntropy= (out.sumEntropy);
    sumOfSquaresVariance= (out.sumOfSquaresVariance);
    sumVariance=(out.sumVariance);
    features_glcm=[];
    features_glcm=[mean(Autocorrelation) mean(clusterProminence) mean(clusterShade) mean(contrast) mean(correlation) mean(differenceEntropy) mean(differenceVariance)...
        mean(dissimilarity) mean(energy) mean(entropy) mean(homogeneity) mean(informationMeasureOfCorrelation1) mean(informationMeasureOfCorrelation2) mean(inverseDifference) mean(maximumProbability)...
        mean(sumAverage) mean(sumEntropy) mean(sumOfSquaresVariance) mean(sumVariance)];
    % Chalkiness
    [pixelCount, grayLevels] = imhist(im);
    thresholdValue = 90; % Choose the threshold value in roder to mask out the background noise.
    % Show the threshold as a vertical red bar on the histogram.
    binaryImage = im > thresholdValue;
    binaryImage = imfill(binaryImage, 'holes');
    MaskedImage = im;
    MaskedImage(~binaryImage) = 0;
    blobMeasurements = regionprops(binaryImage, im, 'all');
    [pixelCount_1, grayLevels_1] = imhist(MaskedImage);
    thresholdValue_1 = 180; %% Choose the threshold that segments the chalkiness
    binary_MaskedImage = MaskedImage > thresholdValue_1;
    binary_MaskedImage = imfill(binary_MaskedImage, 'holes');
    blobMeasurements_subblobs = regionprops(binary_MaskedImage, im, 'all');
    Sum_Subblobs_Perimeter = sum([blobMeasurements_subblobs(1:end).Perimeter]);
    Sum_Blobs_Perimeter = sum([blobMeasurements(1:end).Perimeter]);
    Percentage_Chalkiness=(Sum_Subblobs_Perimeter/Sum_Blobs_Perimeter)*100;
    
    temp=[];
    temp=[fea_hog Percentage_Chalkiness];
    test_fea(ii,:)=temp;
    save test_fea test_fea
end