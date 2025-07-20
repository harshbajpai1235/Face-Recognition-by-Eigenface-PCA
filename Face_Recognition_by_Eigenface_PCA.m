imageFolder = '/Users/harshbajpai/Documents/MATLAB/FaceRecognition_Data/';
trainingFolder = fullfile(imageFolder, 'ALL/');
testFolder = fullfile(imageFolder, 'FA/');
FBFolder = fullfile(imageFolder,'FB/');
trainingFiles = dir(fullfile(trainingFolder, '*.tif'));
%Finding the number of images
M = length(trainingFiles);
% First read one image to determine dimensions
firstImg = imread(fullfile(trainingFolder, trainingFiles(1).name));
imgSize = numel(firstImg); % Total number of pixels in each image
[rows, cols] = size(firstImg);
% Initialize matrix with correct dimensions
trainingImages = zeros(imgSize, M); % Each column will be a vectorized image
for i = 1:M
img = imread(fullfile(trainingFolder, trainingFiles(i).name)); % reading images one by one
trainingImages(:, i) = double(img(:)); % and here as you said i'm vectorizing and storing images
end
X = trainingImages; % vectorized
meanFace = mean(X, 2); % computing mean face across all images
CenterImages = X - repmat(meanFace, 1, M); % Subtracting mean from each image
% calculating smaller covariance matrix as earlier the multiplication was
% not possible for eigen faces
% Using smaller covariance matrix (MxM instead of imgSize x imgSize)
smallCovMatrix = CenterImages' * CenterImages;
% Solve for Eigen values of the smaller matrix
[smallEigenVectors, eigenValues] = eig(smallCovMatrix);
% Sort eigenvalues and corresponding eigenvectors
[sortedEigenValues, indices] = sort(diag(eigenValues), 'descend');
sortedSmallEigenVectors = smallEigenVectors(:, indices);
% Select top K eigenvectors
K = 7; % Cannot have more than M-1 meaningful eigenfaces
topSmallEigenVectors = sortedSmallEigenVectors(:, 1:K);
% Convert small eigenvectors to actual eigenfaces in image space
eigenFaces = CenterImages * topSmallEigenVectors;
% Normalize eigenfaces
for i = 1:K
eigenFaces(:,i) = eigenFaces(:,i) / norm(eigenFaces(:,i));
end
trainingFeatures = eigenFaces' * CenterImages;
% Loading test images ( % these are FA images )
testFiles = dir(fullfile(testFolder, '*.tif'));
numTestImages = length(testFiles);
testImages = zeros(imgSize, numTestImages);
testLabels = zeros(numTestImages,1);
for ti = 1:numTestImages
timg = imread(fullfile(testFolder, testFiles(ti).name));
testImages(:, ti) = double(timg(:));
filename = testFiles(ti).name;
testLabels(ti) = str2double(regexp(filename, '\d+', 'match', 'once'));
end
testImageCentered = zeros(size(testImages));
for j = 1:numTestImages
testImageCentered(:, j) = testImages(:,j) - meanFace;
end
weightstest = eigenFaces' * testImageCentered;
% loading FB images (for testing)
FBFiles = dir(fullfile(FBFolder, '*.tif'));
numFBImages = length(FBFiles);
FBImages = zeros(imgSize, numFBImages);
FBLabels = zeros(numFBImages, 1);
for i = 1:numFBImages
img = imread(fullfile(FBFolder, FBFiles(i).name));
FBImages(:, i) = double(img(:));
% Extracting person number from filename
filename = FBFiles(i).name;
FBLabels(i) = str2double(regexp(filename, '\d+', 'match', 'once'));
end
CenteredFB = zeros(size(FBImages));
for j = 1:numFBImages
CenteredFB(:,j) = FBImages(:, j) - meanFace;
end
weightsFB = eigenFaces' * CenteredFB;
predictedLabels = zeros(numFBImages, 1);
totalConfidence = 0;
numCorrectMatch = 0;
for ti = 1:numFBImages
testImageVector = FBImages(:, ti);
testImageCentered = CenteredFB(:, ti);
testImageProjected = weightsFB(:, ti);
%Calculating distances manually
distances = zeros(1, numTestImages);
for i = 1:numTestImages
diff = testImageProjected - weightstest(:,i);
distances(i) = sqrt(sum(diff.^2));
end
[minDistance, bestMatchIndex] = min(distances);
predictedLabels(ti)= testLabels(bestMatchIndex);
confidence = 1/(1+minDistance);
totalConfidence = totalConfidence + confidence;
if FBLabels(ti) == testLabels(bestMatchIndex)
numCorrectMatch = numCorrectMatch +1;
end
%Visualization
figure;
%Original test image
subplot(1,3,1);
testImg = reshape(FBImages(:, ti), rows, cols);
imshow(uint8(testImg));
titleStr = sprintf('Test Image FB%d', FBLabels(ti));
title(titleStr);
%Mean Face
subplot(1,3,2);
meanImg = reshape(meanFace, rows, cols);
imshow(uint8(meanImg));
title('Mean Face');
%Recognized Face
subplot(1,3,3);
recognizedFace = reshape(testImages(:, bestMatchIndex), rows, cols);
imshow(uint8(recognizedFace));
matchedTitleStr = sprintf('Matched: FA%d', testLabels(bestMatchIndex));
title(matchedTitleStr);
%Print recognition result
fprintf('Test Image: FB%d/n', FBLabels(ti));
fprintf('Best Match: FA%d\n', testLabels(bestMatchIndex));
fprintf('Recognition Confidence: %.2f%%\n', confidence *100);
end
%Calculate overall accuracy
accuracy = numCorrectMatch/numFBImages;
accuracy_percentage = accuracy * 100;
averageConfidence = (totalConfidence/numFBImages) *100;
fprintf('Overall Recongnition Rate: %d/%d (%.2f%%)\n', numCorrectMatch, numFBImages, accuracy_percentage);
fprintf('Average Recognition Confidence: %.2f%%\n', averageConfidence);
%display eigenfaces
figure;
for i = 1:min(K,9)
subplot(3,3,i);
eigenImg = reshape(eigenFaces(:, i), rows, cols);
imshow(eigenImg, []);
title(['Eigenface' num2str(i)]);
end