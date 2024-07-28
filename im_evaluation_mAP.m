function [mAP, pre, rec, auc] = im_evaluation_mAP(opts, fdata)
    % im_evaluation_map: run full evaluation pipeline on specified data.
    % input:
    %   eval: include evaluation modular parameters, struct type
    %   opts: include im system global parameters, struct type
    % output:
    %   mAP: return the mAP value, double type
    
    data = im_evaluation_load(opts, fdata);
    
    img_features_data = data.f_data;
    img_features_name = data.f_name;
    img_query_data = data.q_data;
    img_query_name = data.q_name;
    img_groundtruth_data = data.gt_data;
    tempstr = string(img_groundtruth_data(:,1));
    
    query_num = length(img_query_name);
    ap = single(zeros(query_num, 4));
%     disp(['Computing the performance on (', num2str(query_num), '):       ']);
    opts_match_metric = opts.match.metric;
    opts_match_qe_positive = opts.match.qe_positive;
    opts_match_qe_negative = opts.match.qe_negative;
    for i = 1:query_num
        img_groundtruth_data2 = img_groundtruth_data;
        this_query_X = img_query_data(i,:);
        this_img_query_name = string(img_query_name{i});
        idx = matches(tempstr, this_img_query_name);
        this_img_groundtruth_data = img_groundtruth_data2(idx,:);
        
        [indexs, ~] = get_nn(this_query_X, img_features_data, opts_match_metric);
        if opts_match_qe_positive > 0
            this_query_X = im_cross_query_expansion(this_query_X, img_features_data, indexs, opts_match_qe_positive);
            [indexs, ~] = get_nn(this_query_X, img_features_data, opts_match_metric);
        end
        if opts_match_qe_negative > 0
            this_query_X_negative = im_cross_query_expansion_negative(this_query_X, img_features_data, indexs, opts_match_qe_negative);
            [indexs_negative, ~] = get_nn(this_query_X_negative, img_features_data, opts_match_metric);
            [indexs, ~] = get_nn2(indexs, indexs_negative, img_features_data);
        end
        
        ap(i, :) = get_ap(indexs, this_img_query_name, img_features_name, this_img_groundtruth_data, opts);
%         disp(['compute the (', num2str(i), '/', num2str(size(img_query_name,2)), ') AP: ', num2str(ap(i,1))]);
%         fprintf(1, '\b\b\b\b\b\b%6d', i);
    end
    mAP = mean(ap(:,1));
    pre = mean(ap(:,2));
    rec = mean(ap(:,3));
    auc = mean(ap(:,4));
%     fprintf(1, '\n');
end

function [indexs, distances] = get_nn(this_query_X, img_features_data, metric)
	% Find the k top indexs and distances of index data vectors from query vector x.
    
    if metric == 1
        % L1 distance (Manhattan distance)
        distances = sum(abs(img_features_data - this_query_X), 2);
        [~, indexs] = sort(distances);
    end
    
    if metric == 2
        % L2 distance (Euclidean distance)
        distances = sum((img_features_data - this_query_X) .^ 2, 2) .^ (1/2);
        [~, indexs] = sort(distances);
    end
    
    if metric == 3
        % Canberra distance
        x = abs(img_features_data - this_query_X);
        y = abs(img_features_data) + abs(this_query_X);
        distances = sum(x ./ (y + 1.0), 2);
        [~, indexs] = sort(distances);
    end
    
    if metric == 4
        % Correlation similarity
        x = mean(this_query_X, 2);
        y = mean(img_features_data, 2);
        xy = sum((img_features_data - y) .* (this_query_X - x), 2);
        xy_ = (sum((img_features_data - y) .^ 2, 2) .^ (1/2)) .* (sum((this_query_X - x) .^ 2, 2) .^ (1/2));
        distances = xy ./ (xy_ + 1e-9);
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 5
        % Cosine similarity
        x = sum(img_features_data .* this_query_X, 2);
        y = (sum(img_features_data .^2, 2) .^ (1/2)) .* (sum(this_query_X .^ 2, 2) .^ (1/2));
        distances = x ./ ( y + 1e-9);
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 6
        % Histogram intersection
       
%         bins = size(img_features_data, 2);
%         level = fix(log2(bins));
%         for i = 1:size(img_features_data, 1)
%             k = level;
%             xy = 0;
%             for j = 1:level+1
%                 x_ = histcounts(this_query_X, 2 .^ k);
%                 y_ = histcounts(img_features_data(i,:), 2 .^ k);
%                 xy_ = sum(min(x_, y_), 2) - xy;
%                 level_(j) = (1 / (2 .^ k)) .* xy_;
%                 xy = xy_;
%                 k = k-1;
%             end
%             distances(i, 1) = sum(level_);
%         end
        
%         x = sum(this_query_X, 2);
%         y = sum(img_features_data, 2);
%         xy = sum(min(this_query_X, img_features_data), 2);
%         distances = 1 - xy ./ x;
        
%         x = this_query_X;
%         y = img_features_data;
%         for i = 1:size(y,1)
%             for j = 1:size(y,2)
%                 if x(1, j) > 0 && y(i, j) > 0
%                     xy(i, j) = min(x(1, j), y(i, j));
%                 elseif x(1, j) < 0 && y(i, j) < 0
%                     xy(i, j) = abs(max(x(1, j), y(i, j)));
%                 else
%                     xy(i, j) = 0;
%                 end
%             end
%         end
%         distances = sum(xy, 2);
        
        distances = sum(min(abs(this_query_X), abs(img_features_data)), 2);
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 7
        % Inner product distance
        distances = img_features_data * this_query_X';
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 8
        % Chebyshev distance
        distances = max(abs(img_features_data - this_query_X), [], 2);
        [~, indexs] = sort(distances, 'ascend');
    end
    indexs = single(indexs);
end

function [indexs, distances] = get_nn2(indexs, indexs_negative, img_features_data)
	% Find the k top indexs and distances of index data vectors from query vector x.

    this_query_X_positive = img_features_data(indexs(1),:);
    this_query_X_negative = img_features_data(indexs_negative(1),:);
    
    d1 = sum((img_features_data - this_query_X_positive).^2, 2);
    d2 = sum((img_features_data - this_query_X_negative).^2, 2);
%     distances =  d1 ./ ((d1.^2 + d2.^2).^(1/2));
    distances =  (d1.^2) ./ (d2.^2);
    
    [~, indexs] = sort(distances);
end

function ap = get_ap(indexs, this_img_query_name, img_features_name, this_img_groundtruth_data, opts)
    rank_file_name = [char(this_img_query_name), opts.file.format_mat];   % get the rank name from indexs
    indexs_name = "";
    for i = 1:length(indexs)
        indexs_name(i) = img_features_name(indexs(i));
    end
%    save([opts.match.rank_path, rank_file_name], 'indexs_name');

    rank_set = indexs_name;                     % compute the ap
    good_set = string(this_img_groundtruth_data{2});
    ok_set = string(this_img_groundtruth_data{3});
    junk_set = string(this_img_groundtruth_data{4});
    datasets = string(this_img_groundtruth_data{5});
       
    gt_num = (numel(good_set) + numel(ok_set));
    old_recall = single(0.0);
    old_precision = single(1.0);
    map = single(0.0);
    tp = single(0);
    this_topn = single(1);
    topn = opts.match.precisiontop;
    precision = single(0.0);
    recall = single(0.0);
    
    if isempty(good_set)
        good_set = "";
    end
    if isempty(ok_set)
        ok_set = "";
    end
    if isempty(junk_set)
        junk_set = "";
    end
    
    for i = 1:length(rank_set)
        if ismember(rank_set(i), junk_set)
            continue;
        end
        if tp == gt_num
            break;
        end
        if ismember(rank_set(i), good_set) || ismember(rank_set(i), ok_set)
            tp = tp + 1;
        end
        this_recall = tp / gt_num;
        this_precision = tp / this_topn;
        if strcmp(datasets, "ukbench")
            map = map + tp;
            tp = 0;
            if i == 4
                break;
            end
        else
            map = map +  (abs(this_recall - old_recall)) * ((this_precision + old_precision) / 2.0);
        end
        old_recall = this_recall;
        old_precision = this_precision;
        this_topn = this_topn + 1;
        if i == topn
            precision = tp / topn;
            recall = tp / gt_num;
        end
    end
    
    topn = gt_num;    
    auc_temp = zeros(1, topn);
%     if ~strcmp(datasets, 'ghim10k')
%         for this_topn = 1:topn
%             tp = 0;
%             for i = 1:this_topn
%                 if ismember(rank_set(i), junk_set)
%                     continue;
%                 end
%                 if ismember(rank_set(i), good_set) || ismember(rank_set(i), ok_set)
%                     tp = tp + 1;
%                 end
%             end
%             if this_topn > 1
%                 auc_p_this = tp / this_topn;
%                 auc_r_this = tp / gt_num;
%                 auc_temp(this_topn) = (auc_p_this + auc_p_old) * (auc_r_this - auc_r_old) / 2;
%             end
%             auc_p_old = tp / this_topn;
%             auc_r_old = tp / gt_num;        
%         end
%     end
    auc = sum(auc_temp);
    
    ap(1) = map;
    ap(2) = precision;
    ap(3) = recall;
    ap(4) = auc;
end
