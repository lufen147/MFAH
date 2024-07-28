function report_eval = im_evaluation(opts, fdata)
    % this file is 2nd step runtime on im project
    % This script file steps as below:
    % get the config globle parameters, struct type and named opts, 
    % and then set evaluation paramers, struct type and named eval, 
    % and then input opts and eval to mAP function, output mAP value, double type.
    % Authors: F. Lu. 2020.

    % clear opts eval;
    opts.file.name = mfilename;    % get this script file name
    
    %% get the im config parameters from im_config.m
    if ~exist('tag', 'var')
        opts = im_config(opts);
    end
    %% set evaluation parameters

    opts.eval.img_groundtruth_data = ['img_groundtruth_data_', opts.datasets.name];

    %% calculate mAP
    [mAP, pre, rec, auc] = im_evaluation_mAP(opts, fdata);
    disp([opts.features.net, '_', opts.features.cross_model, '_', opts.features.pipeline_model, '_', opts.datasets.name, '_', num2str(opts.features.dimension), ...
        ': mAP: ', num2str(mAP, '%.4f'),...
        '; Top', num2str(opts.match.precisiontop),...
        ': Precision: ', num2str(pre, '%.4f'),...
        '; Recall: ', num2str(rec, '%.4f'), ...
        '; AUC: ', num2str(auc, '%.4f')]);              % output and display

    %% generate test report
    if ~exist('report_eval.mat', 'file')
        r = 1;
    else
        load('report_eval');
        r = numel(report_eval);
        r = r + 1;
    end
    report_eval(r).frame = opts.features.net;
    report_eval(r).method = opts.features.cross_model;
    report_eval(r).whitening = opts.features.pipeline_model;
    report_eval(r).datasets = opts.datasets.name;
    report_eval(r).dimension = opts.features.dimension;
    report_eval(r).mAP = mAP;
    report_eval(r).Presicion = pre;
    report_eval(r).Recall = rec;
    report_eval(r).AUC = auc;
    report_eval(r).totaltime = toc;
    report_eval(r).datetime = datetime;
    save('report_eval', 'report_eval');

    report_eval = struct2table(report_eval);
    writetable(report_eval, 'report_eval.csv');

    % disp('successed to save the eval report'); 
end