clear all;
close all;
rand_list = [30:10:200];
max_hd_layers = 3;
num_epochs= 100;

generations = 10;
fit_surv = 3;
lucky_surv = 2;
pop_strength = 10;
mutation_prob = 0.3;
population = cell(1,pop_strength);
Acc_matrix = zeros(generations,pop_strength);
Cost_conv = zeros(2,num_epochs);


%Load datasets
load('ExYaleB_Hog_feat.mat');
load('ExYaleB_Hog_Labels.mat');
%Variables inside feat variable: xF, for feature vectors. imSize, size of each
%image. nFaces: Number of individuals (Number of classes).
%Variables inside Labels variable: ExYaleB_HoG_Labels data.

%LNum contains index of first face in each class: [1:64:64*37+1]
LNum = [1:size(xF,2)/nFaces:(floor(size(xF,2)/nFaces))*(nFaces-1) + 1];

%Prepare a sequence of labels for comparison with the output of neural
%network. Labels are numbered starting from 0 instead of 1 to make the code
%from MNIST reusable without least to no modification
Labels = [0:nFaces-1].';
%Create indicator matrix for Labels
%LD = dummyvar(Labels);

%Partition data sets for training, validation and testing

%Prepare training and validation sets from first 32 images of each person
TrainSet = [];
LTrainSet = [];
for ii = 1:nFaces
    TrainSet = [TrainSet xF(:,LNum(ii):LNum(ii)+(floor(size(xF,2)/nFaces)/2)-1)];
    LTrainSet = [LTrainSet; Labels(ii)*ones(floor(size(xF,2)/nFaces)/2,1)];
end

%Prepare test set from remaining 32 images for each of the label
TestSet = [];
LTestSet = [];
for ii = 1:nFaces
    TestSet = [TestSet xF(:,LNum(ii)+ (floor(size(xF,2)/nFaces)/2):LNum(ii)+size(xF,2)/nFaces-1)];
    LTestSet = [LTestSet; Labels(ii)*ones(floor(size(xF,2)/nFaces)/2,1)];
end
C_m = TestSet;
C_l = LTestSet;



%Initialize population with random number of hidden layers and number of 
%neurons in each layer selected randomly from rand_list
for ii = 1:pop_strength
    population{ii} = rand_list(randperm(length(rand_list),randi(max_hd_layers)));
end

%Initialize puplation score
pop_score = zeros(1,pop_strength);
%Indicator function to avoid retraing of an already trained member
pop_train = zeros(1,pop_strength);

%Train and breed population
for ii = 1:generations
    ii
    for jj = 1:pop_strength
        if(pop_train(jj) == 0) %Train only if training alrady not done
            %train data and get the fitness score in terms of accuracy
            pop_score(jj) = train_data(TrainSet,LTrainSet,nFaces,population{jj},num_epochs);
            pop_train(jj) = 1;
        end
    
    end
    disp('All membes nets trained. Starting cross-breeding now');
    %Sort population wrt fitness score
    [pop_score,idx] = sort(pop_score,'descend');
    dummy_pop = cell(1,pop_strength);
    for jj = 1:length(idx)
        dummy_pop{jj} = population{idx(jj)};
    end
    population = dummy_pop;
    Acc_matrix(ii,:) = pop_score;
    
    %Find the fittest members and few lucky ones to see the light of the
    %next day. All others would be replaced with new children
    idx = randperm(pop_strength-fit_surv,lucky_surv) + fit_surv;
    dummy_pop = cell(1,lucky_surv);
    for jj = 1:lucky_surv
        dummy_pop{jj} = population{idx(jj)};
    end
    
    for jj = 1:lucky_surv
        population{fit_surv+jj} = dummy_pop{jj};
    end
    
    %Start cross breeding
    jj = fit_surv + lucky_surv + 1;
    while jj <= pop_strength
        %Select parents randomly
        idx = randperm(fit_surv+lucky_surv,2);
        L1 = length(population{idx(1)});
        L2 = length(population{idx(2)});
        if L2 > L1
            Par1 = [population{idx(1)} zeros(1,L2-L1)];
            Par2 = population{idx(2)};
        elseif L1 > L2
            Par1 = population{idx(1)};
            Par2 = [population{idx(2)} zeros(1,L1-L2)];
        else
            Par1 = population{idx(1)};
            Par2 = population{idx(2)};
        end
        %Crosslink and make children
        Child1 = zeros(1,length(Par1));
        Child2 = zeros(1,length(Par1));
        for k = 1:length(Par1)
            if randi(2) == 1
                Child1(k) = Par1(k);
                Child2(k) = Par2(k);
            else
                Child1(k) = Par2(k);
                Child2(k) = Par1(k);
            end
        end
        population{jj} = Child1([find(Child1)]);
        pop_score(jj) = 0;
        pop_train(jj) = 0;
        %Perform random mutation in the child
        if rand > mutation_prob
            idx = randperm(length(population{jj}),1);
            population{jj}(idx) = rand_list(randperm(length(rand_list),1));
        end
        if jj+1 <= pop_strength
            jj = jj+1;
            population{jj} = Child2([find(Child2)]);
            pop_score(jj) = 0;
            pop_train(jj) = 0;
            if rand > mutation_prob
                idx = randperm(length(population{jj}),1);
                population{jj}(idx) = rand_list(randperm(length(rand_list),1));
            end
        end
        jj = jj+1;
    end
    disp('New generation created');
end

%Optimization using genetic algorithm complete
%Find final accuracy scores



    
        
            
            
                
    
    
    
        

        
        
        