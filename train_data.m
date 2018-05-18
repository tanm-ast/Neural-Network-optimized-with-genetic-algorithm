function Accu = train_data(TrainSet,LTrainSet,nFaces,hid_nN,num_epochs)

%k-fold cross validation is to be done
k_fold = 5;
%Parameter in the sigmoid/tanh function
a = 1.5;
%Learning rate
mu = 0.05;
beta1 = 0.9;
epsilon = 10^(-8);

%Mini-batch size and maximum number of epochs
mbatch = 60;


%Choice of activation function
%1 => Hyperbolic tangent y = tanh(x/a);
%2 => Sigmoid  y = 1 ./ (1 + exp(-a*x));
%Higher the value of a steeper is the transition
act_f_choice = 1;

Q = zeros(1,num_epochs);

nN = [size(TrainSet,1),hid_nN,nFaces];
nLayer = length(nN);

size_dataset = length(LTrainSet);
%Number of mini batches
numbatch = floor((size_dataset - floor(size_dataset/k_fold))/mbatch);
samples = [1:size_dataset];
%Sample elements for validation set randomly
val_set = randperm(size_dataset,floor(size_dataset/k_fold));

%Build validation and training sets

%Cross validation set
cv_data = TrainSet(:,val_set);
cv_labels = LTrainSet(val_set);
%Training set
train_set_im = TrainSet(:,setdiff(samples,val_set));
train_set_labels = LTrainSet(setdiff(samples,val_set));
%Randomize the order of contents in training set
perm = randperm(size_dataset - floor(size_dataset/k_fold));
train_set_im = train_set_im(:,perm);
train_set_labels = train_set_labels(perm);

F_m = [cv_data train_set_im];
F_l = [cv_labels; train_set_labels];

Conv = [];
for n = 1:k_fold
    tic
    n
    %Initialize all weights and biases
    %First rows are for biases
    %Each column corresponds to weights going to a neuron in the next level
    W = cell(1,nLayer-1);
    v = cell(1,nLayer-1);
    Y = cell(1,nLayer-1);
    Y{1} = zeros(nN(1)+1,mbatch);
    del = cell(1,nLayer-1);
    m_1 = cell(1,nLayer-1);
    m_2 = cell(1,nLayer-1);
    %Initialize weights 
    for ii = 1:nLayer-1
         W{ii} = rand(nN(ii)+1,nN(ii+1))/(nN(ii)+1);
         m_1{ii} = zeros(nN(ii)+1,nN(ii+1));
         m_2{ii} = zeros(nN(ii)+1,nN(ii+1));
         v{ii} = zeros(nN(ii+1),mbatch);
         Y{ii+1} = zeros(nN(ii+1)+1,mbatch);
         del{ii} = zeros(nN(ii+1),mbatch);
    end
    
    %Build new validation and training sets
    val_set = (n-1)*length(cv_labels)+1:n*length(cv_labels);
    cv_data = F_m(:,val_set);
    cv_labels = F_l(val_set);
    train_set_im = F_m(:,setdiff(samples,val_set));
    train_set_labels = F_l(setdiff(samples,val_set));
    %Randomize training data
    rand_idx = randperm(length(train_set_labels));
    train_set_im = train_set_im(:,rand_idx);
    train_set_labels = train_set_labels(rand_idx);
    %Training Starts
    J = 1000;
    q = 1;
    
    while (q < num_epochs)
        %fprintf('epoch %d\n',q);
        
        for z = 1:numbatch
            z;
            if z == numbatch
                %To handle the scenario when mini-batch size is not a 
                %factor of total training set size
                train_im_z = train_set_im(:,(z-1)*mbatch+1:end);
                train_l_z = train_set_labels((z-1)*mbatch+1:end);
            else
                train_im_z = train_set_im(:,(z-1)*mbatch+1:z*mbatch);
                train_l_z = train_set_labels((z-1)*mbatch+1:z*mbatch);
            end
            
            x = train_im_z;
            Y{1} = [ones(1,size(x,2)); x];
            lab = train_l_z;

            %Forward computations follow
            for ii = 1:nLayer-1
                v{ii} = W{ii}.' * Y{ii};
                if act_f_choice == 1
                    Y{ii+1} = tanh_act(v{ii},a);
                elseif act_f_choice == 2
                    Y{ii+1} = smoid(v{ii},a);
                end
                    
                %First element of the vector corresponds to the bias 
                Y{ii+1} = [ones(1,size(x,2)); Y{ii+1}];
            end

            Y_lab = zeros(nN(nLayer),size(x,2));
            for z_l = 1:size(x,2)
                Y_lab(lab(z_l)+1,z_l) = 1;
            end

            %Compute squared error
            e = Y{nLayer}(2:end,:) - Y_lab;


            %Cost function
            J = 0.5*sum(sum(e.^2))/size(x,2);

            %Backward computations follow                     
            for ii = nLayer-1:-1:1
                if act_f_choice == 1
                    del{ii} = e .* diff_tanh_act(v{ii},a);
                elseif act_f_choice == 2
                    del{ii} = e .* diff_smoid(v{ii},a);                    
                end
                e = W{ii}(2:end,:) * del{ii};
            end

            %Weights update
            for ii = 1:nLayer-1
                del_j = Y{ii} * del{ii}.';
                %Update 1st moment estimate and remove bias
                m_1{ii} = (beta1*m_1{ii} + (1 - beta1)*(del_j.^2));
                %Determine step to take at this iteration
                del_w = -(mu/size(x,2))*(del_j./(sqrt(m_1{ii} + epsilon)));
                W{ii} = W{ii} + del_w;
                
            end

        end
        Q(q) = Q(q) + J;
        q = q+1;
    end

    
    %Cross validation
    In_i = [ones(1,size(cv_data,2)); cv_data];

    %Forward computations follow
    for ii = 1:nLayer-1
        if act_f_choice == 1
            In_i = [ones(1,size(cv_data,2)); tanh_act(W{ii}.' * In_i,a)];
        elseif act_f_choice == 2
            In_i = [ones(1,size(cv_data,2)); smoid(W{ii}.' * In_i,a)];
        end
    end

    [~, t] = max(In_i(2:end,:));
    t = t.' - 1;
    c = (cv_labels == t);
    %Find errors
    Conv = [Conv, sum(c)/length(c)];
    %Epoch_Con = [Epoch_Con, q];
    Q = Q/5;
    %display('Validation complete');
    %display('Building new training and validation set');
    toc

end

Accu = mean(Conv);
Cost_value = Q(end);

display('Cross-Validation complete');
%return;
