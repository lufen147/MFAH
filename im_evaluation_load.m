function data = im_evaluation_load(opts, fdata)
    % im_evaluation_load: run and load data for evaluating.
    % input:
    %   eval: include evaluation modular parameters, struct type
    %   opts: include im system global parameters, struct type
    % output:
    %   data: return the data sets, struct type, include:
    %       f_data: return the img features data, n * p double type
    %       f_name: return the img features name, n * p double type
    %       q_data: return the img query data, n * p double type
    %       q_name: return the img query name, n * p double type
    %       gt_data: return the ground data, n * 5 cell type   
    
   %% load features data and query data
    
    img_features_data = fdata.AGGF;
    img_features_name = fdata.name;
    path = [opts.features.path_query, opts.file.format_common, opts.file.format_txt];
    [img_query_data, img_query_name, ~] = im_evaluation_load_query(img_features_data, img_features_name, path, opts); 
    
    %% load groundtruth data
    if exist([opts.run.data_temp, opts.eval.img_groundtruth_data], 'file')
        load([opts.run.data_temp, opts.eval.img_groundtruth_data]);
    else
        path = opts.datasets.gt_path;
        img_groundtruth_data = im_evaluation_load_groundtruth(img_query_name, path, opts);
%         save([opts.run.data_temp, opts.eval.img_groundtruth_data], 'img_groundtruth_data');
    end
    
    %% return data
    data.f_data = img_features_data;
    data.f_name = img_features_name;
    data.q_data = img_query_data;
    data.q_name = img_query_name;
    data.gt_data = img_groundtruth_data;
end
