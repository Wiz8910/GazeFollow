%test contains test_bbox
%              test_eyes
%              test_gaze
%              test_meta
%              test_path
load('test_annotations')
%train contains train_bbox
%               train_eyes
%               train_gaze
%               train_meta
%               trainf_path
load('train_annotations')
trainSaveFile = 'trainData.mat';
testSaveFile = 'testData.mat';


trainGazeStartX = [];
trainGazeEndX = [];
trainGazeStartY = [];
trainGazeEndY = [];
trainIm =[];
WIDTH = 256;%256;
HEIGHT = 256;%256;

thresh=0.1;
numIms=1000;
i=1;
while i<=numIms%size(train_eyes,2)
    im = imread(train_path{i});
    %im = im(:,:,1);
    if(size(im,3)==3)
		%if you want a colorspace convert here
        %im = rgb2gray(im);
        im = imresize(im,[WIDTH,HEIGHT]);
        trainIm= [trainIm im(:)];
        trainGazeStartX = [trainGazeStartX train_eyes{i}(1)];
        trainGazeEndX = [trainGazeEndX train_gaze{i}(1)];
        trainGazeStartY = [trainGazeStartY train_eyes{i}(2)];
        trainGazeEndY = [trainGazeEndY train_gaze{i}(2)];
    else
    	numIms = numIms+1;
    end
    i=i+1;
end
%normalize images
trainIm = double(trainIm)/double(max(trainIm(:)));
trainIm = double(trainIm); %every column in X is one vectorized input image
rowsDeviating = [];
%feature_ind = find(cov(X,2)>thresh);
stdX =std(trainIm');%std(X');%
%stdX = diag(stdX)';
rowsDeviating =[1,find(stdX>thresh)];
trainIm = trainIm(rowsDeviating,:);


trainIm = [ones(1,size(trainIm,2)); trainIm];
save(trainSaveFile,'trainIm','trainGazeStartX','trainGazeEndX','trainGazeStartY','trainGazeEndY','-v7.3');
clearvars trainIm trainGazeStartX trainGazeStartY trainGazeEndX trainGazeEndY;
disp('Done with one');
numData = 0;
testGazeStartX = [];
testGazeEndX = [];
testGazeStartY = [];
testGazeEndY = [];
testIm =[];
numTestIms= 1000;
i=1;
while i<=numTestIms%size(test_eyes,2)
    im = imread(test_path{i});
    %im = im(:,:,1);
    if(size(im,3)==3)
        %im = rgb2gray(im);
        im = imresize(im,[WIDTH,HEIGHT]);
        testIm= [testIm im(:)];
        testGazeStartX = [testGazeStartX test_eyes{i}(1)];
        testGazeEndX = [testGazeEndX test_gaze{i}(1)];
        testGazeStartY = [testGazeStartY test_eyes{i}(2)];
        testGazeEndY = [testGazeEndY test_gaze{i}(2)];
    else
        numTestIms = numTestIms+1;
    end
    i=i+1;
end
%normalize images
testIm = double(testIm)/double(max(testIm(:)));
testIm = double(testIm); %every column in X is one vectorized input image
testIm = testIm(rowsDeviating,:);

testIm = [ones(1,size(testIm,2)); testIm];
save(testSaveFile,'testIm','testGazeStartX','testGazeEndX','testGazeStartY','testGazeEndY','-v7.3');


method = 'dualLinear';
load(trainSaveFile)%;,trainIm,trainGazeStartX,trainGazeEndX,trainGazeStartY,trainGazeEndY);
load(testSaveFile);%,testIm,testGazeStartX,testGazeEndX,testGazeStartY,testGazeEndY);

[avgError, startX,startY,inferredX,inferredY] = ClassifyImages(trainIm,trainGazeStartX,trainGazeEndX,trainGazeStartY,trainGazeEndY,...
                                testIm,testGazeStartX,testGazeEndX,testGazeStartY,testGazeEndY,method);
sum = inferredX+inferredY;
disp(avgError);
imsToShow = 1;
for i=1:imsToShow%size(test_eyes,2)
    eyeLoc = test_eyes{i};
    gazeLoc = test_gaze{i};
    fig = imread(test_path{i});
    figure,imshow(fig)
    %figure; 
    %imshow(fig,[]); title('red: detection, green: groundtruth');
    hold on
    width = size(fig,2);
    height = size(fig,1);
    plot([width*eyeLoc(1) width*gazeLoc(1,1)],[height*eyeLoc(2) height*gazeLoc(1,2)],'Color','r','LineWidth',4);
    plot([width*eyeLoc(1) width*inferredX(i)],[height*eyeLoc(2) height*inferredY(i)],'Color','g','LineWidth',4);
    %imagesc(fig);
    %line([eyeLoc(1) gazeLoc(1,1)],[eyeLoc(2) gazeLoc(1,2)],'Color','r','LineWidth',1000);
    %hold off;
end
%}