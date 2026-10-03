%% Initialize
clear; close all; clc;

%% 1. Load and prepare dataset
datasetPath = 'C:\Users\SS\Downloads\Dataset_Organized';
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display class distribution
labelCounts = countEachLabel(imds);
numClasses = height(labelCounts);
figure;
bar(labelCounts.Count);
xticklabels(labelCounts.Label);
xtickangle(90);
title('Class Distribution');
ylabel('Number of Images');

%% 2. Data preprocessing with face detection
function img = preprocessFace(inputImg)
    persistent faceDetector;
    if isempty(faceDetector)
        faceDetector = vision.CascadeObjectDetector();
    end
    
    % Convert to grayscale if needed
    if size(inputImg, 3) == 3
        grayImg = rgb2gray(inputImg);
    else
        grayImg = inputImg;
    end
    
    % Detect face
    bbox = step(faceDetector, grayImg);
    
    % Crop to face region
    if ~isempty(bbox)
        margin = 30;
        x1 = max(1, bbox(1)-margin);
        y1 = max(1, bbox(2)-margin);
        x2 = min(size(inputImg,2), bbox(1)+bbox(3)+margin);
        y2 = min(size(inputImg,1), bbox(2)+bbox(4)+margin);
        img = inputImg(y1:y2, x1:x2, :);
    else
        img = inputImg;
    end
    
    % Convert to RGB if single channel
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    end
    
    % Resize and normalize
    img = imresize(img, [224 224]);
    img = single(img);
    img = (img - 127.5) / 127.5;
end

% Preview preprocessing
figure;
sampleImg = readimage(imds, 1);
subplot(1,2,1); imshow(sampleImg); title('Original');
subplot(1,2,2); imshow(preprocessFace(sampleImg)); title('Processed');

%% 3. Create balanced dataset
targetCount = 60;
balancedFiles = {};
balancedLabels = {};

for i = 1:numel(labelCounts.Label)
    classFiles = imds.Files(imds.Labels == labelCounts.Label(i));
    
    % Oversample if needed
    if numel(classFiles) < targetCount
        repeats = ceil(targetCount/numel(classFiles));
        classFiles = repmat(classFiles, repeats, 1);
    end
    
    % Undersample if needed
    classFiles = classFiles(1:min(targetCount, numel(classFiles)));
    
    balancedFiles = [balancedFiles; classFiles];
    balancedLabels = [balancedLabels; repmat(labelCounts.Label(i), numel(classFiles), 1)];
end

balancedImds = imageDatastore(balancedFiles, 'Labels', categorical(balancedLabels));

% Display balanced class distribution
labelCounts = countEachLabel(balancedImds);
numClasses = height(labelCounts);
figure;
bar(labelCounts.Count);
xticklabels(labelCounts.Label);
xtickangle(90);
title('Class Distribution');
ylabel('Number of Images');

%% 4. Data splitting and augmentation
[imdsTrain, imdsVal] = splitEachLabel(balancedImds, 0.8, 'randomized');

augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...
    'RandScale', [0.9 1.1], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augImdsTrain = augmentedImageDatastore([224 224], imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsVal = augmentedImageDatastore([224 224], imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% 5. Network architecture - Base Model (ResNet50)
net = resnet50;
lgraph = layerGraph(net);

% Remove classification layers
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% Get class names
classNames = categories(imdsTrain.Labels);

%% 6. Warmup Training (without ArcFace)
warmupLayers = [
    fullyConnectedLayer(512, 'Name', 'fc_embed', 'WeightLearnRateFactor', 10)
    batchNormalizationLayer('Name', 'bn_embed')
    reluLayer('Name', 'relu_embed')
    fullyConnectedLayer(numClasses, 'Name', 'fc_final')
    softmaxLayer()
    classificationLayer()
];

lgraphWarmup = addLayers(lgraph, warmupLayers);
lgraphWarmup = connectLayers(lgraphWarmup, 'avg_pool', 'fc_embed');

% Warmup training options
warmupOptions = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augImdsVal, ...
    'ValidationFrequency', 50, ...
    'ExecutionEnvironment', 'gpu', ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Warmup training
[netWarmup, warmupInfo] = trainNetwork(augImdsTrain, lgraphWarmup, warmupOptions);

%% 7. ArcFace Implementation 
embeddingSize = 128;
margin = 0.5;
scale = 64;

% Create fresh layer graph from warmup network
layers = netWarmup.Layers;

% Remove last 3 classification layers
layers = layers(1:end-3);

% Add new ArcFace layers
layers = [
    layers
    fullyConnectedLayer(embeddingSize, 'Name', 'fc_embed_arcface', 'WeightLearnRateFactor', 10)
    batchNormalizationLayer('Name', 'bn_embed_arcface')
    l2NormalizationLayer()
    ArcFaceTransformLayer(numClasses, embeddingSize, margin, scale, 'arcface_transform')
    softmaxLayer('Name', 'softmax_arcface')
    classificationLayer('Name', 'classification_arcface')
];

% Create new network
lgraph = layerGraph(layers);

% ArcFace training options
arcfaceOptions = trainingOptions('adam', ...
    'InitialLearnRate', 0.0005, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'ValidationData', augImdsVal, ...
    'ValidationFrequency', 100, ...
    'ExecutionEnvironment', 'gpu', ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10);

% Train with ArcFace
[netArcface, arcfaceInfo] = trainNetwork(augImdsTrain, lgraph, arcfaceOptions);

%% 8. Evaluation
[YPred, scores] = classify(netArcface, augImdsVal);
figure;
confusionchart(imdsVal.Labels, YPred);
title('Confusion Matrix');

%% 9. Save models
save('faceRecognitionNet_warmup.mat', 'netWarmup', 'warmupInfo', '-v7.3');
save('faceRecognitionNet_arcface.mat', 'netArcface', 'arcfaceInfo', '-v7.3');
save('faceEmbeddingNet.mat', 'netArcface', '-v7.3');

%% Helper Functions
function [isMatch, score] = verifyFace(net, img1, img2, threshold)
    embedding1 = activations(net, preprocessFace(img1), 'l2_norm_arcface', 'OutputAs', 'columns');
    embedding2 = activations(net, preprocessFace(img2), 'l2_norm_arcface', 'OutputAs', 'columns');
    
    embedding1 = embedding1 ./ vecnorm(embedding1, 2, 1);
    embedding2 = embedding2 ./ vecnorm(embedding2, 2, 1);
    
    score = dot(embedding1, embedding2);
    isMatch = score > threshold;
end