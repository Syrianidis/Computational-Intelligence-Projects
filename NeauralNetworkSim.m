function [NN,traTPR,tesTPR,traMSE,tesMSE,TR,trInit,traFinalOut,tsInit,tesFinalOut]=NNsimR(numHidLays,epVal,lrVal,mcVal)
% [NN,traTPR,tesTPR,traMSE,tesMSE,TR,trInit,traFinalOut,tsInit,tesFinalOut]=NNsim(epVal,lrVal,mcVal,numHidLays)
% 
% DESCRIPTION
%   Create and simulate a feed-forward backpropagation Neural Network (NN)
%   using for training the training function gradient descent with momentum
% 
% INPUT
%   epVal       is the maximum number of epochs the NN's training allowed
%   lrVar       is the learning rate of the NN
%   mcVal       is the momentum constant of NN's training
%   numHidLays  is the number of hidden layers of the NN
% 
% OUTPUT
%   NN          is the neural network created and simulated
%   traTPR      is the true positive rate for the training set
%   tesTPR      is the true positive rate for the test set
%   traMSE      is the mean squared error for the training set
%   tesMSE      is the mean squared error for the test set
%   TR          is the training record of the NN's training process
%   trInit      is the initial output vector of the training set
%   traFinalOut is the output vector of the NN for the training set
%   tsInit      is the initial output vector of the test set
%   tesFinalOut is the output vector of the NN for the test set
% 
% EXAMPLE
%   The following example creates and simulates a NN with 64 inputs,
%   37 hidden layers, 10 output layers, learning rate .1, 
%   momentum constant .5, and maximum number of training epochs 1000 
% 
%   [NN,traTPR,tesTPR]=NNsim(37,1e3,.1,.5);
% 
%   The variable NN is the neural network, and the variables traTPR, tesTPR
%   contain a value from range [.4 .5] (40%-50%), because of randomness
%   
% SEE ALSO
% nnstart, newff, init, traingdm, sim, mapminmax, mse, plotperform,
% plottrainstate, plotregression, ploterrhist

if nargin<4
    mcVal=.1;
end
if nargin<3
    lrVal=.9;
end
if nargin<2
    epVal=1e3;
end
if nargin<1
    numHidLays=37;
end

%% Q 1.1

% we load the data given to us
% NOTE: the files 'tra.mat' and 'tes.mat' must be moved to the same file as
% the working directory - use pwd( ) to find your working directory
load('tra');
load('tes');

%% Q 1.2

% % % traSet=tra(:,1:end-1)-mean2(tra(:,1:end-1));
% % % traSet=traSet/max(traSet(:));
% % % tesSet=tes(:,1:end-1)-mean2(tes(:,1:end-1));
% % % tesSet=tesSet/max(tesSet(:));

% we normalize the data in [-1,1]
traSet=mapminmax(tra(:,1:end-1));
tesSet=mapminmax(tes(:,1:end-1));

%% Q 1.3

% we initialize the output vector as a matrix where all elements are zeros
% and we will use it to allocate to each sample a one (1) to the
% corresponding element of the output vector
% meaning that if in the ith sample the output vector contains a 5 then the
% output matrix will contain the row 0 0 0 0 1 0 0 0 0 0
traOut=zeros(size(tra,1),1+range(tra(:,end)));
tesOut=zeros(size(tes,1),1+range(tes(:,end)));

% we use the values of the output vector as indexes incremented by one to
% match Matlab's array referencing in order to create the output matrix
for i=1:size(traOut,1)
    traOut(i,1+tra(i,end))=1;
end
for i=1:size(tesOut,1)
    tesOut(i,1+tes(i,end))=1;
end
% as the example given, we flip the output matrix in order to match index
% of 1's to the values of output vector right to left
traOut=fliplr(traOut);
tesOut=fliplr(tesOut);

%% Q 1.4

% we concatenate the modified data and then we randomly permutate the
% records
traR=horzcat(traSet,traOut);
index=randperm(length(traR));
trQ=traR(index,:);
trInit=tra(index,end);

tesR=horzcat(tesSet,tesOut);
index=randperm(length(tesR));
tsQ=tesR(index,:);
tsInit=tes(index,end);

%% Q 2

% we separate out data in order to use them in our problem accordingly
trainIn=trQ(:,1:size(traSet,2));
trainTgt=trQ(:,size(traSet,2)+1:size(trQ,2));

testIn=tsQ(:,1:size(tesSet,2));
testTgt=tsQ(:,size(tesSet,2)+1:size(tsQ,2));

X=trainIn;
T=trainTgt;

% inputs will represent the min (left column) and the max (right column)
% values of each feature of each set
inputs=[-ones(size(X,2),1) ones(size(X,2),1)];
% outputs will represent the output layers of the NN
outputs=size(testTgt,2);

% we create a feed-forward backpropagation NN with 64 inputs, numHidLays
% hidden layers, 10 output layers, logsig transfer function, and traingdm
% training function
NN=newff(inputs,[numHidLays outputs],{'logsig', 'logsig'},'traingdm'); % manual page 640
% we initialize the NN
NN=init(NN); % manual page 549

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % view(NN)

% % % neural_inputs=zeros(num_of_inputs,2);
% % %         neural_inputs(:,1)=-1;
% % %         neural_inputs(:,2)=1;
% % %       %  net = newff(neural_inputs,[20 1],{'logsig', 'purelin'},'trainlm');
% % %         net=newfit(training_new',output2',floor((num_of_inputs+1)/2));
% % %         net = init(net);
% % %         net.divideFcn='divideblock';
% % %         net.divideParam.trainRatio = 0.8;
% % %         net.divideParam.valRatio = 0.2;
% % % %net.divideParam.testRatio = 0;
% % %         net.trainParam.goal = 0;
% % %         net.trainParam.show = NaN;
% % %         net.trainParam.showWindow=0;

% we modify the training parameters we want to alter
% maximum epochs default set to 1000 (nargin)
NN.trainParam.epochs=epVal;
% learning rate default set to .9 (nargin)
NN.trainParam.lr=lrVal;
% momentum constant default set to .1 (nargin)
NN.trainParam.mc=mcVal;
% maximum validation checks set to 10
NN.trainParam.max_fail=10;
% minimum gradient set to 1e-10
NN.trainParam.min_grad=1e-10;

% we train the NN using the traingdm training function
[NN,TR]=traingdm(NN,X',T'); % manual page 857

%% Q 3

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% for plots, we remove the interpretor
set(0,'defaulttextinterpreter','none')

% we simulate the NN for the training set
YYtra=sim(NN,X'); % manual page 769 and so on
Y=YYtra;
% the outputs are given weights, so we keep the strongest (set 1) and we
% ignore all the rest (set 0)
for i=1:length(Y)
    temp=Y(:,i);
    temp(temp==max(temp))=1;
    temp(temp~=max(temp))=0;
    Y(:,i)=temp;
end
% because we flipped the original sets in order to match our preferences,
% we flip it in order to correct it
Ytra=flipud(Y);

% because of the combination of functions mod( ) and find( ), the value 9
% is replaced by -1, so we set every value -1 to 9
temp=mod(find(Ytra==1),10);
temp(temp==0)=10;
traFinalOut=temp-1;

% we concatenate the initial outputs with final outputs and calculate our
% results
CHECKtra=horzcat(trInit,traFinalOut);

% we calculate the true positive rate for the training set
qqtra=CHECKtra(:,1)-CHECKtra(:,2);
traTPR=numel(qqtra(qqtra==0))/length(CHECKtra);

% we calculate the mean squared error for the training set
% traMSE=mse(NN,Ytra,T'); % manual page 627
traMSE=norm(traFinalOut-trInit)/numel(trInit);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

X=testIn;
T=testTgt;

% we simulate the NN for the test set
YYtes=sim(NN,X'); % manual page 769 and so on
Y=YYtes;
% the outputs are given weights, so we keep the strongest (set 1) and we
% ignore all the rest (set 0)
for i=1:length(Y)
    temp=Y(:,i);
    temp(temp==max(temp))=1;
    temp(temp~=max(temp))=0;
    Y(:,i)=temp;
end
% because we flipped the original sets in order to match our preferences,
% we flip it in order to correct it
Ytes=flipud(Y);

% because of the combination of functions mod( ) and find( ), the value 9
% is replaced by -1, so we set every value -1 to 9
temp=mod(find(Ytes==1),10);
temp(temp==0)=10;
tesFinalOut=temp-1;

% we concatenate the initial outputs with final outputs and calculate our
% results
CHECKtes=horzcat(tsInit,tesFinalOut);

% we calculate the true positive rate for the test set
qqtes=CHECKtes(:,1)-CHECKtes(:,2);
tesTPR=numel(qqtes(qqtes==0))/length(CHECKtes);

% we calculate the mean squared error for the test set
% tesMSE=mse(NN,Ytes,T'); % manual page 627
tesMSE=norm(tesFinalOut-tsInit)/numel(tsInit);
