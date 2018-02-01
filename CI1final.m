% Computational Inteligence I - Final Script
% 
% 5-fold Cross-Validations and Visualizations of Results

% keep('Results');
clear all;
close all;clc

epVal=10.^linspace(2,4,3);
lrVal=linspace(.1,.9,3);
mcVal=linspace(.1,.9,3);

Results=cell(28,12);
Results{1,1}='epochs';
Results{1,2}='learning rate';
Results{1,3}='momentum constant';
Results{1,4}='TPR train';
Results{1,5}='TPR test';
Results{1,6}='MSE train';
Results{1,7}='MSE test';
Results{1,8}='Training Record';
Results{1,9}='Init Out train';
Results{1,10}='Final Out train';
Results{1,11}='Init Out test';
Results{1,12}='Final Out test';
count=2;

X=[];

for q=1:5
    for i=1:numel(epVal)

        for j=1:numel(lrVal)

            for z=1:numel(mcVal)

                [NN,Results{count,4},...
                    Results{count,5},...
                    Results{count,6},...
                    Results{count,7},...
                    Results{count,8},...
                    Results{count,9},...
                    Results{count,10},...
                    Results{count,11},...
                    Results{count,12}]=NNsimR(37,epVal(i),lrVal(j),mcVal(z));
                Results{count,1}=epVal(i);
                Results{count,2}=lrVal(j);
                Results{count,3}=mcVal(z);
                count=count+1;

                clc

            end

        end

    end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % THIS IS FOR ME! % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    v=genvarname('X',who); % BEWARE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    eval([v '=Results']);  % the first X must be defined before!!!!!!!!!!!!
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % THIS IS FOR ME! % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    
end

traTPR=mean(horzcat(vertcat(X1{2:28,4}),vertcat(X2{2:28,4}),vertcat(X3{2:28,4}),vertcat(X4{2:28,4}),vertcat(X5{2:28,4})),2);
tesTPR=mean(horzcat(vertcat(X1{2:28,5}),vertcat(X2{2:28,5}),vertcat(X3{2:28,5}),vertcat(X4{2:28,5}),vertcat(X5{2:28,5})),2);

traMSE=mean(horzcat(vertcat(X1{2:28,6}),vertcat(X2{2:28,6}),vertcat(X3{2:28,6}),vertcat(X4{2:28,6}),vertcat(X5{2:28,6})),2);
tesMSE=mean(horzcat(vertcat(X1{2:28,7}),vertcat(X2{2:28,7}),vertcat(X3{2:28,7}),vertcat(X4{2:28,7}),vertcat(X5{2:28,7})),2);


QQ=zeros(27,5);
for i=1:5
    temp1=eval(['X',num2str(i)]);
    temp2=temp1;
    for j=2:28
        temp3=temp1{j,9};
        temp4=temp2{j,10};
        QQ(j-1,i)=norm(temp3-temp4)/numel(temp3); 
    end 
end
traMSEnew=mean(QQ,2);

QQ=zeros(27,5);
for i=1:5
    temp1=eval(['X',num2str(i)]);
    temp2=temp1;
    for j=2:28
        temp3=temp1{j,11};
        temp4=temp2{j,12};
        QQ(j-1,i)=norm(temp3-temp4)/numel(temp3);
    end
end

tesMSEnew=mean(QQ,2);
