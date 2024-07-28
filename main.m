% Authors: F. Lu, G-H. Liu, from 2021 to 2024.

%% 1 Load general config
warning off; clear; tic;
opts = im_config();
opts.datasets.prefixname1 = 'AGGF1_';
opts.run.data_temp = './data_temp/';

% A_data = ["Oxford5K", "Paris6K", "Holidays_upright", "Flickr100K", "revisitop1m"];
A_data = ["Oxford5K", "Paris6K", "Holidays_upright"];
B_net = ["vgg16"];

%% 2 Calculating raw Features

for a_i = 1:size(A_data,2)
    for b_i = 1:size(B_net,2)
        opts.datasets.name = lower(A_data(a_i));
        opts.features.net  = lower(B_net(b_i));
        opts = im_config(opts);
        
       % % 2.1 set dataset parameter
      
        filepatch = strcat(opts.datasets.image_path, '*.jpg');
        isfile = dir(filepatch);
        if numel(isfile) == 0
            filepatch = strcat(opts.datasets.image_path, '/*/*.jpg');
        end
        isfile = dir(filepatch);
        if numel(isfile) == 0
            filepatch = strcat(opts.datasets.image_path, '/*/*.png');
        end
                
        filename = dir(filepatch);
        file_num = size(filename, 1);

        % % 2.2 load pre-trained network
        if ~isempty(opts.features.netmodelpath) && exist(opts.features.netmodelpath, "file")
            net = importdata(opts.features.netmodelpath);
            if isstruct(net)
                net = net.net;
            end
        else
            net = eval(opts.features.net);
        end
        if ismember(opts.features.net, ["vgg16"])
            layer1 = 'pool5';
            dim1 = 512;
            layer2 = 'fc7';
            dim2 = 4096;
            layer3 = layer1;
        end
        
        if opts.features.inputsize == 1, insz = [1024, 896, 3]; end
        if opts.features.inputsize == 0, insz = net.Layers(1).InputSize; end
        
        % % 2.3 aggregate and save raw feature
       % % 2.3.1 set aggregation parameter
        [cn1, cn2, cn3] = deal(6, 4, 4);
        [CSC, CSO, CSI, CSS] = deal(cn1 * cn2 * cn3, 60, 64, 32);
        Hcnum = CSC + CSO + CSI;
        
        LowF = single(zeros(file_num, 4 * Hcnum));
        MidF = single(zeros(file_num, dim1));
        HighF = single(zeros(file_num, dim2));
        ContF = single(zeros(file_num, 4 * Hcnum + dim1 + dim2));
        [CMF, RCMF, SMF, SCMF, MLF, DMLHF, PMLHF] = deal(MidF);
        [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11] = deal(single(0));
        
        name_list = "";

        % % 2.3.2 prepair to start aggregating
        disp([char(datetime), ' aggregating from ', char(opts.datasets.name), ' using ', char(opts.features.net), ' on (', num2str(file_num), '):        ']);
        
        datasetname = opts.datasets.name;
        for i = 1:file_num
           imdata = imread([filename(i).folder, '/', filename(i).name]);
           imfilename = filename(i).name;
            
           % % 2.3.3 pre-process the size of image
            if size(imdata,3) == 1
                imdata = cat(3,imdata,imdata,imdata);
            end
            img = single(imdata);
            [h, w, ~] = size(img);
            
            img1 = imresize(img, 112/min(h,w));
            
            img_resize = imresize(img, 1024/min(h,w));

            % % 2.3.4 get images name list
            name_split = strsplit(imfilename, {'.'});
            name = name_split(1);
            name_list(i) = name{1};

            % % 2.3.5 get feature representation
            try
                X = activations(net, img_resize, layer1, 'OutputAs', 'channels');
            catch
                X = activations(net, img_resize, layer1, 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');                        
            end
            X1 = X;
            
            try
                X = activations(net, img_resize, layer2, 'OutputAs', 'channels');
            catch
                X = activations(net, img_resize, layer2, 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
            end
            X2 = X;
            
            % MFAH (Multi-Feature Aggregation Histogram) model
            [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11] = Agg_model({X1, X2, img1}, {cn1, cn2, cn3, CSO, CSI, CSS});
            
            LowF(i,:)  = H1;
            MidF(i,:)  = H2;
            HighF(i,:) = H3;
            ContF(i,:) = H4;
            CMF(i,:)   = H5;
            RCMF(i,:)  = H6;
            SMF(i,:)   = H7;
            SCMF(i,:)  = H8;
            MLF(i,:)   = H9;
            DMLHF(i,:) = H10;
            PMLHF(i,:) = H11;
            
            fprintf(1,'\b\b\b\b\b\b\b%7d', i);

        end
        % % 2.3.6 gather various raw feature and save to .mat
        toc
        [s_name,s_feature] = deal("name", "Feature");
        AGGF.(s_name) = name_list;
        [AGGF.(s_feature)(1).Dname, AGGF.(s_feature)(1).Descriptor] = deal('LowF', LowF);
        [AGGF.(s_feature)(2).Dname, AGGF.(s_feature)(2).Descriptor] = deal('MidF', MidF);
        [AGGF.(s_feature)(3).Dname, AGGF.(s_feature)(3).Descriptor] = deal('HighF', HighF);
        [AGGF.(s_feature)(4).Dname, AGGF.(s_feature)(4).Descriptor] = deal('ContF', ContF);
        [AGGF.(s_feature)(5).Dname, AGGF.(s_feature)(5).Descriptor] = deal('CMF', CMF);
        [AGGF.(s_feature)(6).Dname, AGGF.(s_feature)(6).Descriptor] = deal('RCMF', RCMF);
        [AGGF.(s_feature)(7).Dname, AGGF.(s_feature)(7).Descriptor] = deal('SMF', SMF);
        [AGGF.(s_feature)(8).Dname, AGGF.(s_feature)(8).Descriptor] = deal('SCMF', SCMF);
        [AGGF.(s_feature)(9).Dname, AGGF.(s_feature)(9).Descriptor] = deal('MLF', MLF);
        [AGGF.(s_feature)(10).Dname, AGGF.(s_feature)(10).Descriptor] = deal('DMLHF', DMLHF);
        [AGGF.(s_feature)(11).Dname, AGGF.(s_feature)(11).Descriptor] = deal('PMLHF', PMLHF);

        save([opts.run.data_temp, opts.datasets.prefixname1, datasetname{1}, '_', opts.features.net{1}], 'AGGF');
    end
end
disp(char(datetime));

%% 3 Post-process and Evaluation
warning off; tic;
opts.run.data_temp = './data_temp/';

% A_data = ["Oxford5K", "Paris6K", "Holidays_upright", "oxford105k", "paris106k", "Holidays_upright101k", "oxford1005k", "paris1006k", "Holidays_upright1001k"];
A_data = ["Oxford5K", "Paris6K", "Holidays_upright"];
% A_data = ["oxford105k", "paris106k", "Holidays_upright101k"];
% A_data = ["oxford1005k", "paris1006k", "Holidays_upright1001k"];
B_net = ["vgg16"];
C_dim = [16, 32, 64, 128, 256, 512];
C_dim = [128, 256, 512];

% pcacw denotes using pca with cross whitening;
% pcadw denotes using pca with differency whitening;
% norm denotes using L2-normalization without using pca;
% D_pipeline_model = ["norm"];
% D_pipeline_model = ["pcacw"];
D_pipeline_model = ["pcadw"];
% D_pipeline_model = ["norm", "pcacw", "pcadw"];

E_feature = [2 3 7 10 8];
E_feature = [8];

isIDF = 1;

for e_i = 1:size(E_feature, 2)
    kd = E_feature(e_i);
    disp(['**************************** H', num2str(kd),' (', char(datetime), ') *************************************************************']);
    for d_i = 1: size(D_pipeline_model, 2)
        if ismember(D_pipeline_model(d_i), ["norm"])
            C_dim2 = C_dim(end);
        else
            C_dim2 = C_dim;
        end
        for a_i = 1:size(A_data, 2)
            for b_i = 1:size(B_net, 2)
                opts.features.pipeline_model = lower(D_pipeline_model(d_i));
                opts.datasets.name = lower(A_data(a_i));
                opts.features.net = lower(B_net(b_i));                

                % 3.1 post-process and save the finnal feature

                KD = kd;     % get/set the last serial number of Feature for image retrieval
                [KD1, KD2] = deal(0, 0);

                if ismember(opts.datasets.name, ["oxford105k", "paris106k", "oxford1005k", "paris1006k", "holidays_upright101k", "holidays_upright1001k"])
                    if ismember(opts.datasets.name, ["oxford105k", "oxford1005k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'oxford5k', '_', opts.features.net{1}]);
                        AGGF1 = temp.AGGF;
                    end
                    if ismember(opts.datasets.name, ["paris106k", "paris1006k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'paris6k', '_', opts.features.net{1}]);
                        AGGF1 = temp.AGGF;
                    end
                    if ismember(opts.datasets.name, ["holidays_upright101k", "holidays_upright1001k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'holidays_upright', '_', opts.features.net{1}]);
                        AGGF1 = temp.AGGF;
                    end
                    if ismember(opts.datasets.name, ["oxford105k", "paris106k", "holidays_upright101k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'flickr100k', '_', opts.features.net{1}]);
                        AGGF1.name = [AGGF1.name, temp.AGGF.name];                    
                        if KD1>0 && KD2>0
                            AGGF1.Feature(KD1).Descriptor = [AGGF1.Feature(KD1).Descriptor; temp.AGGF.Feature(KD1).Descriptor];
                            AGGF1.Feature(KD2).Descriptor = [AGGF1.Feature(KD2).Descriptor; temp.AGGF.Feature(KD2).Descriptor];
                        else
                            AGGF1.Feature(KD).Descriptor = [AGGF1.Feature(KD).Descriptor; temp.AGGF.Feature(KD).Descriptor];
                        end
                    end
                    if ismember(opts.datasets.name, ["oxford1005k", "paris1006k", "holidays_upright1001k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'revisitop1m', '_', opts.features.net{1}]);
                        AGGF1.name = [AGGF1.name, temp.AGGF.name];                    
                        if KD1>0 && KD2>0
                            AGGF1.Feature(KD1).Descriptor = [AGGF1.Feature(KD1).Descriptor; temp.AGGF.Feature(KD1).Descriptor];
                            AGGF1.Feature(KD2).Descriptor = [AGGF1.Feature(KD2).Descriptor; temp.AGGF.Feature(KD2).Descriptor];
                        else
                            AGGF1.Feature(KD).Descriptor = [AGGF1.Feature(KD).Descriptor; temp.AGGF.Feature(KD).Descriptor];
                        end
                    end
                    clear temp;
                else
                    temp = load([opts.run.data_temp, opts.datasets.prefixname1, opts.datasets.name{1}, '_', opts.features.net{1}]);
                    AGGF1 = temp.AGGF;
                end


                if KD1>0 && KD2>0
                    AGGF = [AGGF1.Feature(KD1).Descriptor, AGGF1.Feature(KD2).Descriptor];
                else
                    AGGF = AGGF1.Feature(KD).Descriptor;
                end
                name = AGGF1.name;
                
                train = AGGF1.Feature(KD).Descriptor;
                
                if ismember(opts.features.pipeline_model, ["pcacw"])
                    if ismember(opts.datasets.name, ["oxford5k", "oxford105k", "oxford1005k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'paris6k', '_', opts.features.net{1}]);
                        train = temp.AGGF.Feature(KD).Descriptor;
                    end
                    if ismember(opts.datasets.name, ["paris6k", "paris106k", "paris1006k", "holidays_upright", "holidays_upright101k", "holidays_upright1001k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'oxford5k', '_', opts.features.net{1}]);
                        train = temp.AGGF.Feature(KD).Descriptor;
                    end
                end
                if ismember(opts.features.pipeline_model, ["pcadw"])
                    if ismember(opts.datasets.name, ["oxford5k", "oxford105k", "oxford1005k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'oxford5k', '_', opts.features.net{1}]);
                        train = temp.AGGF.Feature(KD).Descriptor;
                    end
                    if ismember(opts.datasets.name, ["paris6k", "paris106k", "paris1006k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'paris6k', '_', opts.features.net{1}]);
                        train = temp.AGGF.Feature(KD).Descriptor;
                    end
                    if ismember(opts.datasets.name, ["holidays_upright", "holidays_upright101k", "holidays_upright1001k"])
                        temp = load([opts.run.data_temp, opts.datasets.prefixname1, 'holidays_upright', '_', opts.features.net{1}]);
                        train = temp.AGGF.Feature(KD).Descriptor;
                    end
                end
                clear temp AGGF1;

                if isIDF == 1
                    AGGF = IDF(AGGF);
                    train = IDF(train);
                end

                if ismember(opts.features.pipeline_model, ["pcadw"])
                    d_ = 1;
                    if d_ == 1
                        % L2 distance (fast version)
                        T = train;
                        train2 = im_cross_normalize(train);
                        [n, ~] = size(train);
                        ab = train2 * train2';
                        a2 = diag(ab);
                        b2 = diag(ab)';
                        dist = sqrt(a2 - 2*ab + b2);
                        clear ab a2 b2;
                        for i = 1:n
                            t1 = dist(:,i);
                            [~, idx] = mink(t1, 2);
                            T(i,:) = train(idx(2), :);
                        end
                        train = train - T;
                        train(train<0) = 0;
                    end

                    if d_ == 2
                        % L2 distance (common version)
                        T = train;
                        train2 = im_cross_normalize(train);
                        [n, ~] = size(train);
                        fprintf('%s get difference whitening data ... %d/       ', datetime, n);
                        for i = 1:n
                            t1 = sum((train2(i,:) - train2) .^ 2, 2) .^ (1/2);
                            [~, idx] = mink(t1, 2);
                            T(i,:) = train(idx(2), :);
                            fprintf('\b\b\b\b\b\b\b%7d', i);
                        end
        			    toc
                        train = train - T;
                        train(train<0) = 0;
                    end
                end

                for c_i = 1:size(C_dim2, 2)
                    dim = C_dim2(c_i);

                    AGGF_ = AGGF;
                    if ismember(opts.features.pipeline_model, ["norm"])
                        AGGF_ = im_cross_normalize(AGGF);
                    end
                    if ismember(opts.features.pipeline_model, ["pcacw", "pcadw"])
                        AGGF_ = PCA_whitening(AGGF, train, dim);
                    end

                    AGGF2.name = name;
                    AGGF2.AGGF = AGGF_;
                    dim = size(AGGF_, 2);
                    
%                     save([opts.run.data_temp, opts.datasets.prefixname2, char(opts.datasets.name)], 'AGGF2');
                    fdata = AGGF2;

                    % 3.2 evaluation
                    
                    opts.features.pipeline_model = char(opts.features.pipeline_model);      % here for cross whitening, one of [none, norm, pca, pca_whitening, pca_whitening_self, pca_whitening_augmentation]
                    opts.features.dimension = dim;
                    opts.datasets.name = char(opts.datasets.name);
                    opts.features.net = char(opts.features.net);                        
                    
                    opts.match.metric = 2;
                    opts.match.qe_positive = 0;
                    opts.match.queryratio = 1;
                    opts.match.precisiontop = 10;
                    report_eval = im_evaluation(opts, fdata);
                end
            end
        end
    end
end
disp(char(datetime));