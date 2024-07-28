function [img_query_data, img_query_name, img_query_image_name] = im_evaluation_load_query(img_features_data, img_features_name, path, opts)
    % im_evaluation_load_query: Given the query image name and features after features processing.
    % input:
    %   img_features_data: features after features processing
    %   img_features_name: corresponding features names
    %   path: query image info directory, string
    % output:
    %   img_query_data: the list of loaded query features, list
    %   img_query_name: corresponding file names without extension, list
    %   img_query_image_name: corresponding image names without extension, list
    
    img_query_data = single([]);
    img_query_name = "";
    img_query_image_name = "";
    img_query_info = dir(path);
    if length(img_query_info) < 1
        extract_query(opts);
        img_query_info = dir(path);
    end
   
%     disp(['load query images feature from ', path, '(total: ', num2str(size(img_query_info,1)), ')      ']);
    for i = 1:size(img_query_info)
        this_img_query_name = split(img_query_info(i).name, '.');
        img_query_name(i) = this_img_query_name{1};
        
        this_img_query_features_name = importdata([img_query_info(i).folder, '/', img_query_info(i).name]);
        if iscell(this_img_query_features_name)
            img_query_image_name(i) = this_img_query_features_name{1};
        else
            img_query_image_name(i) = this_img_query_features_name;
        end
        
        j = img_features_name == img_query_image_name(i);
        this_query_info_X = img_features_data(j, :);
        img_query_data(i,:) = this_query_info_X;
        
%         fprintf(1,'\b\b\b\b\b\b%6d',fix(i));
    end
%     fprintf(1,'\n');
end
function extract_query(opts)
    path = [opts.datasets.gt_path, opts.file.format_common];
    if ismember(opts.datasets.name, ["oxford5k", "paris6k", "oxford105k", "paris106k", "oxford1005k", "paris1006k"])
        path = [opts.datasets.gt_path, opts.file.format_common, '_query', opts.file.format_txt];
    end
    if ismember(opts.datasets.name, ["holidays", "holidays_upright", "holidays_upright101k", "holidays_upright1001k", "ukbench"])
        path = [opts.datasets.gt_path, opts.file.format_common, opts.file.format_jpg];
    end
    img_datasets_groundtruth = dir(path);    % get the list of the images file information to img_datasets
    img_datasets_groundtruth_num = size(img_datasets_groundtruth,1);  % count the size of datasets to img_datasets_num
%     disp(['extract query images features file from ', opts.datasets.gt_path, '(total: ', num2str(img_datasets_groundtruth_num), ')      ']);
    if ismember(opts.datasets.name, ["holidays", "holidays_upright", "holidays_upright101k", "holidays_upright1001k"])
        for i=1:img_datasets_groundtruth_num
            this_img_name = split(img_datasets_groundtruth(i).name, '.');
            this_img_filename = this_img_name{1};       
            if mod(str2double(this_img_filename), 100) == 0
               dlmwrite([opts.features.path_query, this_img_filename, opts.file.format_txt], this_img_filename, 'delimiter', '', 'newline', 'pc');
            end
%             fprintf(1,'\b\b\b\b\b\b%6d', fix(i));
        end
    end
    
    if ismember(opts.datasets.name, ["oxford5k", "paris6k", "oxford105k", "paris106k", "oxford1005k", "paris1006k"])
        for i=1:img_datasets_groundtruth_num
            this_img_path = [img_datasets_groundtruth(i).folder, '/', img_datasets_groundtruth(i).name];     % form one txt file path
            this_img_name_raw = importdata(this_img_path);
            this_img_filename_split = split(img_datasets_groundtruth(i).name, '.');    % read this image name and split to file name and format name
            this_img_filename = strrep(this_img_filename_split{1}, '_query', '');
            this_img_name = strrep(this_img_name_raw.textdata{1}, 'oxc1_', '');            
            dlmwrite([opts.features.path_query, this_img_filename, opts.file.format_txt], this_img_name, 'delimiter', '', 'newline', 'pc');
%             fprintf(1,'\b\b\b\b\b\b%6d', fix(i));
        end
    end        
end
