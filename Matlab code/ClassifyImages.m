function [sumOfErrors, startX, startY, targetX, targetY] = ClassifyImages(trainIm,trainGazeStartX,trainGazeEndX,trainGazeStartY,trainGazeEndY,...
                                      testIm,testGazeStartX,testGazeEndX,testGazeStartY,testGazeEndY,method)
lambda=.1;
temp = trainIm'*trainIm;
thingToInverse = (temp*temp + lambda*eye(size(trainIm,2)));
sumOfErrors=0;
%no longer inferring start labels

psiStartX = trainIm*(thingToInverse \ (temp*trainGazeStartX'));
startX=psiStartX'*testIm;% psiStartX'*testIm'*testIm;%'*phi;
sumOfErrors= sum(abs(testGazeStartX(:)-startX(:)))/size(testGazeStartX,2);

psiStartY = trainIm*(thingToInverse \ (temp*trainGazeStartY'));
startY =psiStartY'*testIm;% psiStartX'*testIm'*testIm;%'*phi;
sumOfErrors= sumOfErrors+sum(abs(testGazeStartY(:)-startY(:)))/size(testGazeStartY,2);

psiEndX = trainIm*(thingToInverse \ (temp*trainGazeEndX'));
targetX =psiEndX'*testIm;% psiStartX'*testIm'*testIm;%'*phi;
sumOfErrors= sumOfErrors+sum(abs(testGazeEndX(:)-targetX(:)))/size(testGazeEndX,2);


psiEndY = trainIm*(thingToInverse \ (temp*testGazeEndY'));
targetY =psiEndY'*testIm;% psiStartX'*testIm'*testIm;%'*phi;
sumOfErrors= sumOfErrors+sum(abs(testGazeEndY(:)-targetY(:)))/size(testGazeEndY,2);

sumOfErrors = sumOfErrors/2;

%{

psiStartY = trainIm*(((trainIm'*trainIm)*(trainIm'*trainIm) + lambda*eye(size(trainIm,2)))...
        \(trainIm'*trainIm*trainGazeStartY'));
testW = psiStartY'*testIm;%'*phi; testIm'
sumOfErrors= sumOfErrors+sum(abs(testGazeStartY(:)-testW(:)))/size(testGazeStartY,2);
psiEndX = ((trainIm'*trainIm)*(trainIm'*trainIm) + lambda*eye(size(trainIm,2)))...
        \(trainIm'*trainIm*trainGazeEndX');
testW = psiEndX'*testIm'*testIm;%'*phi;
sumOfErrors= sumOfErrors+sum(abs(testGazeEndX(:)-testW(:)))/size(testGazeEndX,2);
psiEndY = ((trainIm'*trainIm)*(trainIm'*trainIm) + lambda*eye(size(trainIm,2)))...
        \(trainIm'*trainIm*testGazeEndY');
testW = psiEndY'*testIm'*testIm;%'*phi;
sumOfErrors= sumOfErrors+sum(abs(testGazeEndY(:)-testW(:)))/size(testGazeEndY,2);

if strcmp(method,'linear') 
    phiStart = (trainIm*trainIm')\(trainIm*w);
    testW = X_test'*phi;
    sumOfErrors= sum(abs(w_true(:)-testW(:)))/size(w_true,1);
elseif( strcmp(method,'slowLinear'))
    phi = pinv(X*X')*(X*w);
    testW = X_test'*phi;
    sumOfErrors= sum(abs(w_true(:)-testW(:)))/size(w_true,1);
elseif strcmp(method,'bayes')
    phi = (X*X'+lambda* eye(size(X,1),size(X,1)))\(X*w);    
    testW = X_test'*phi;
    sumOfErrors= sum(abs(w_true(:)-testW(:)))/size(w_true,1);
elseif strcmp(method,'nonLinear')
    %create z from x^1 + x^2 +x^3
    Z=[];
    for i=2:4
       Z = [Z; X.^(i-1)]; 
    end
    phi = (Z*Z' + lambda*eye(size(Z,1)))\(Z*w);
    Z_test = [];
    for i=2:4
       Z_test = [Z_test; X.^(i-1)]; 
    end
    testW = Z_test'*phi;
    sumOfErrors= sum(abs(w_true(:)-testW(:)))/size(w_true,1);
elseif strcmp(method,'dualLinear')
    psi = ((X'*X)*(X'*X) + lambda*eye(size(X,2)))\(X'*X*w);
    testW = psi'*X_test'*X_test;%'*phi;
    sumOfErrors= sum(abs(w_true(:)-testW(:)))/size(w_true,1);
else
    sumOfErrors=0;
%}
end