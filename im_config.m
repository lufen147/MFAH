function opts = im_config(opts)
    % config file of project on your assign data sets
    % hyper parameters config
    % Authors: F. Lu. 2020.

    s_data  = ["datasets", "features", "match"];
    s_basic = ["file", "run", "extract", "param"];
    s = [s_basic, s_data];

    for i = 1:length(s)
        opts.(s{i}).temp = 'none';  % setting placeholder of structure type
    end

    opts.file.tempdir = tempdir;
    opts.file.root = fileparts(mfilename('fullpath'));  % get this project's root, opts, a new define struct type
    root = opts.file.root;          % define a simply variable for using below
    opts.file.name = mfilename;     % get this script file name, not include suffix name
    current_dir = pwd;              % get current info
    current_folder = regexp(current_dir,'\\','split');
    current_folder = current_folder{end};
    current_name = regexp(current_folder,'_','split');
    current_name = current_name{end};

    opts.file.format_txt = '.txt';  % config txt file format, noted that "."
    opts.file.format_jpg = '.jpg';  % config jpg file format, noted that "."
    opts.file.format_mat = '.mat';  % config mat file format, noted that "."
    opts.file.format_npy = '.npy';  % config npy file format, noted that "."
    opts.file.format_dat = '.dat';  % config dat file format, noted that "."
    opts.file.format_cvs = '.cvs';  % config cvs file format, noted that "."
    opts.file.format_common = '*';  % config the images name, * is any name

    if ~isfield(opts.run, 'data_temp'),opts.run.data_temp = './data_temp/'; end       % generate temp mat data
    if ~exist(opts.run.data_temp, 'dir'), mkdir(opts.run.data_temp); end

    if ~isfield(opts.run, 'datapath'),opts.datasets.datapath = '../../data/'; end       % config the data file path
    if ~exist(opts.datasets.datapath, 'dir'), mkdir(opts.datasets.datapath); end

    if ~exist([opts.datasets.datapath, '/datasets/'], 'dir'), mkdir([opts.datasets.datapath, '/datasets/']); end
    if ~exist([opts.datasets.datapath, '/features/'], 'dir'), mkdir([opts.datasets.datapath, '/features/']); end
    if ~exist([opts.datasets.datapath, '/networks/'], 'dir'), mkdir([opts.datasets.datapath, '/networks/']); end

    datapath = opts.datasets.datapath;        % define a simply variable for using below
    datasetslist = ["oxford5k", "paris6k", "holidays_upright", "oxford105k", "paris106k", ...
        "flickr100k", "revisitop1m", "oxford1005k", "paris1006k", "holidays_upright101k", "holidays_upright1001k"];
    for i = 1:length(datasetslist)
        opts.path(i).datasetname = char(datasetslist(i));     % setting meta info of data
    end
    % opts.path(1).datasetinfo.datasets.image_path = fullfile(fileparts(root), 'datasets', 'Oxford5K', 'oxbuild_images');
    opts.path(1).datasetinfo.datasets.image_path = [datapath, '/datasets/Oxford5K/oxbuild_images/'];  % config the images datasets orgin path
    opts.path(1).datasetinfo.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];    % config the groundtruth images orgin path

    opts.path(2).datasetinfo.datasets.image_path = [datapath, '/datasets/Paris6K/paris_images/'];
    opts.path(2).datasetinfo.datasets.gt_path = [datapath, '/datasets/Paris6K/gt_files_120310/'];

    opts.path(3).datasetinfo.datasets.image_path = [datapath, '/datasets/Holidays_upright/images/'];
    opts.path(3).datasetinfo.datasets.gt_path = [datapath, '/datasets/Holidays_upright/images/'];

    opts.path(4).datasetinfo.datasets.image_path = [datapath, '/datasets/Oxford5K/oxbuild_images/'];
    opts.path(4).datasetinfo.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];

    opts.path(5).datasetinfo.datasets.image_path = [datapath, '/datasets/Paris6K/paris_images/'];
    opts.path(5).datasetinfo.datasets.gt_path = [datapath, '/datasets/Paris6K/gt_files_120310/'];

    % opts.path(6).datasetinfo.datasets.image_path = [datapath, '/datasets/Flickr100K/oxc1_100k/'];
    opts.path(6).datasetinfo.datasets.image_path = ['I:', '/dataset/100K/'];
    opts.path(6).datasetinfo.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];
    
    % opts.path(7).datasetinfo.datasets.image_path = [datapath, '/datasets/revisitop1m/r1m_images/'];
    opts.path(7).datasetinfo.datasets.image_path = ['I:', '/dataset/revisitop1m/'];
    opts.path(7).datasetinfo.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];
    
    opts.path(8).datasetinfo.datasets.image_path = [datapath, '/datasets/Oxford5K/oxbuild_images/'];
    opts.path(8).datasetinfo.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];
    
    opts.path(9).datasetinfo.datasets.image_path = [datapath, '/datasets/Paris6K/paris_images/'];
    opts.path(9).datasetinfo.datasets.gt_path = [datapath, '/datasets/Paris6K/gt_files_120310/'];
    
    opts.path(10).datasetinfo.datasets.image_path = [datapath, '/datasets/Holidays_upright/images/'];
    opts.path(10).datasetinfo.datasets.gt_path = [datapath, '/datasets/Holidays_upright/images/'];
    
    opts.path(11).datasetinfo.datasets.image_path = [datapath, '/datasets/Holidays_upright/images/'];
    opts.path(11).datasetinfo.datasets.gt_path = [datapath, '/datasets/Holidays_upright/images/'];

    for i = 1:length(datasetslist)
        temp = char(datasetslist(i));
        opts.path(i).datasetinfo.features.path = [datapath, '/features/', temp, '/pool5/'];              % config the images feature save path    
        opts.path(i).datasetinfo.features.path_query = [datapath, '/features/', temp, '/queries/'];      % config the query images feature save path
        opts.path(i).datasetinfo.features.path_cropquery = [datapath, '/features/', temp, '/pool5_crop_queries/'];
        opts.path(i).datasetinfo.features.path_fc = [datapath, '/features/', temp, '/fc7/'];
        opts.path(i).datasetinfo.features.path_fc_cropquery = [datapath, '/features/', temp, '/fc7_crop_queries/'];
        opts.path(i).datasetinfo.match.rank_path = [datapath, '/features/', temp, '/rank_file/'];
    end

    if ~isfield(opts.datasets, 'name'), opts.datasets.name = 'oxford5k'; end        % config datasets name, one of [oxford5k, paris6k, roxford5k, rparis6k, oxford105k, paris106k, holidays, ukbench, flickr100k]
    if ~isfield(opts.datasets, 'eachclassnum'), opts.datasets.eachclassnum = 100; end        % config each class images of datasets

    for i = 1:length(opts.path)
        if strcmp(opts.datasets.name, opts.path(i).datasetname), path_id = i; end                    % get current dataset path id in the meta info data
    end
    
%     if ~exist([datapath, '/features/', opts.datasets.name, '/'], 'dir'), mkdir([datapath, '/features/', opts.datasets.name, '/']); end
%     if ~exist(opts.path(path_id).datasetinfo.features.path, 'dir'), mkdir(opts.path(path_id).datasetinfo.features.path); end
%     if ~exist(opts.path(path_id).datasetinfo.features.path_cropquery, 'dir'), mkdir(opts.path(path_id).datasetinfo.features.path_cropquery); end
    if ~exist(opts.path(path_id).datasetinfo.features.path_query, 'dir'), mkdir(opts.path(path_id).datasetinfo.features.path_query); end
%     if ~exist(opts.path(path_id).datasetinfo.features.path_fc, 'dir'), mkdir(opts.path(path_id).datasetinfo.features.path_fc); end
%     if ~exist(opts.path(path_id).datasetinfo.features.path_fc_cropquery, 'dir'), mkdir(opts.path(path_id).datasetinfo.features.path_fc_cropquery); end
    if ~exist(opts.path(path_id).datasetinfo.match.rank_path, 'dir'), mkdir(opts.path(path_id).datasetinfo.match.rank_path); end
    
    for i = 1:length(s_data)
        f = fieldnames(opts.path(path_id).datasetinfo.(s_data{i}));
        for j = 1:length(f)
            opts.(s_data{i}).(f{j}) = opts.path(path_id).datasetinfo.(s_data{i}).(f{j});  % add the current dataset info of the meta info data
        end
    end

    if ~isfield(opts.run, 'batchsize'), opts.extract.batchsize = 1; end     % config the batch images number input to CNN while extract feature, option of [1, 64, 128, 256]
    if ~isfield(opts.features, 'query_crop'), opts.features.query_crop = 0; end        % config the query image extract form, value 1 is crop, value 0 is full image (not crop)
    if ~isfield(opts.features, 'isloading'), opts.features.isloading = 0; end
    if ~isfield(opts.features, 'inputsize'), opts.features.inputsize = 0; end    
    if ~isfield(opts.features, 'issave'), opts.features.issave = 0; end

    if ~isfield(opts.features, 'net'), opts.features.net = 'vgg16'; end   % config net model frame, one of [vgg16, caffe, matconvnet, matconvnet_dag]
    temp = char(opts.features.net);
    if opts.features.issave == 1
        if ~exist([opts.features.path, '/', temp, '/'], 'dir'), mkdir([opts.features.path, '/', temp, '/']); end
        if ~exist([opts.features.path_fc, '/', temp, '/'], 'dir'), mkdir([opts.features.path_fc, '/', temp, '/']); end
        if ~exist([opts.features.path_cropquery, '/', temp, '/'], 'dir'), mkdir([opts.features.path_cropquery, '/', temp, '/']); end
        if ~exist([opts.features.path_fc_cropquery, '/', temp, '/'], 'dir'), mkdir([opts.features.path_fc_cropquery, '/', temp, '/']); end
    end

    opts.features.netmodelpath = [];
    if strcmp(opts.features.net, 'alexnet'), opts.features.netmodelpath = [datapath, '/networks/imagenet/alexnet.mat']; end
    if strcmp(opts.features.net, 'vgg16'), opts.features.netmodelpath = [datapath, '/networks/imagenet/vgg16.mat']; end
    if strcmp(opts.features.net, 'vgg19'), opts.features.netmodelpath = [datapath, '/networks/imagenet/vgg19.mat']; end
    if strcmp(opts.features.net, 'googlenet'), opts.features.netmodelpath = [datapath, '/networks/imagenet/googlenet.mat']; end
    if strcmp(opts.features.net, 'resnet18'), opts.features.netmodelpath = [datapath, '/networks/imagenet/resnet18.mat']; end
    if strcmp(opts.features.net, 'resnet50'), opts.features.netmodelpath = [datapath, '/networks/imagenet/resnet50.mat']; end
    if strcmp(opts.features.net, 'resnet101'), opts.features.netmodelpath = [datapath, '/networks/imagenet/resnet101.mat']; end
    if strcmp(opts.features.net, 'densenet201'), opts.features.netmodelpath = [datapath, '/networks/imagenet/densenet201.mat']; end
    if strcmp(opts.features.net, 'mobilenetv2'), opts.features.netmodelpath = [datapath, '/networks/imagenet/mobilenetv2.mat']; end    
    if strcmp(opts.features.net, 'efficientnetb0'), opts.features.netmodelpath = [datapath, '/networks/imagenet/efficientnetb0.mat']; end    
    if strcmp(opts.features.net, 'inceptionv3'), opts.features.netmodelpath = [datapath, '/networks/imagenet/inceptionv3.mat']; end

    if ~isfield(opts.features, 'net_layer'), opts.features.net_layer = 'pool5'; end             % config images feature extracted from which net layter
    if ~isfield(opts.features, 'dimension'), opts.features.dimension = 128; end              % config the images feature extracted dimension
    if ~isfield(opts.features, 'cross_model'), opts.features.cross_model = current_name; end       % config calculate cross model, one of [mhdf], mhdf: mid- and high deep feature
    if ~isfield(opts.features, 'pipeline_model'), opts.features.pipeline_model = 'none'; end     % config pipeline model such as Dimension reduction model, one of [none, norm, pca, pca_whitening, pca_whitening_self, pca_relja, pca_whitening_relja, pca_pairs]

    if ~isfield(opts.match, 'qe_positive'), opts.match.qe_positive = 0; end     % config image retrieval query expansion positive top R, if do not use query expansion that put R=0
    if ~isfield(opts.match, 'qe_negative'), opts.match.qe_negative = 0; end     % config image retrieval query expansion negative bottom R, if do not use that put R=0
    if ~isfield(opts.match, 'metric'), opts.match.metric = 2; end               % config the metric (measure) method. option: L1 Manhattan distance:1, L2 Euclidean:2, Canberra distance:3, Correlation similarity:4, Cosine similarity:5, Histogram intersection:6, Inner product distance:7, Chebyshev distance:8,  
    if ~isfield(opts.match, 'precisiontop'), opts.match.precisiontop = 12; end  % config the precision (including recall, auc) top N images.
    if ~isfield(opts.match, 'queryratio'), opts.match.queryratio = 0.1; end     % config each query ratio of datasets

    save('opts', 'opts');       % save and use for some module loading
end
