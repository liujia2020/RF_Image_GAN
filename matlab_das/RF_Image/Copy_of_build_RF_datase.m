DataRoot = 'G:\Data_0110_RFdata';
ManifestDir = fullfile(DataRoot, '_manifest');

manifest = build_RF_dataset_manifest_v1(DataRoot, ManifestDir);

% 先做一个每类 1 个文件的 pilot manifest，用来后面小规模生成样本
pilot_manifest = make_pilot_manifest_v1( ...
    manifest, 1, fullfile(ManifestDir, 'RF_manifest_pilot_1perclass.csv'));

% 也可以之后生成每类 10 个文件的小测试集
% pilot10_manifest = make_pilot_manifest_v1( ...
%     manifest, 10, fullfile(ManifestDir, 'RF_manifest_pilot_10perclass.csv'));
function manifest = build_RF_dataset_manifest_v1(DataRoot, ManifestDir)

% ============================================================
% build_RF_dataset_manifest_v1
%
% Purpose:
%   Build a file-level manifest for RF learning sample generation.
%
% Data structure assumed:
%   DataRoot/
%       Simu_Data/
%       Muscle_Data/
%       Carotid_Data/
%       Phantom_Data/
%
% Category mapping:
%   Simu_Data    -> simu_point
%   Muscle_Data  -> muscle
%   Carotid_Data -> carotid
%   Phantom_Data -> phantom
%
% Split strategy:
%   For each category, use at most 100 files for balanced learning.
%   Split selected files as:
%       70% train
%       15% val
%       15% test
%
%   Extra Simu_Data files are kept in the manifest as:
%       split = extra_psf_holdout
%       use_for_learning = false
%
% Output columns:
%   file_id
%   category
%   split
%   use_for_learning
%   num_patches
%   file_path
%   file_name
%   folder_name
%   local_index
% ============================================================

    if nargin < 2 || isempty(ManifestDir)
        ManifestDir = fullfile(DataRoot, '_manifest');
    end

    if ~exist(ManifestDir, 'dir')
        mkdir(ManifestDir);
    end

    rng(20260521);  % fixed seed for reproducible split

    % ------------------------------------------------------------
    % Category definitions
    % ------------------------------------------------------------
    category_defs = {
        'Simu_Data',    'simu_point', 100;
        'Muscle_Data',  'muscle',     100;
        'Carotid_Data', 'carotid',    100;
        'Phantom_Data', 'phantom',    100;
    };

    patches_per_file = 5;

    train_ratio = 0.70;
    val_ratio   = 0.15;
    % test_ratio is implicit

    % ------------------------------------------------------------
    % Prepare table columns
    % ------------------------------------------------------------
    file_id_list = {};
    category_list = {};
    split_list = {};
    use_for_learning_list = [];
    num_patches_list = [];
    file_path_list = {};
    file_name_list = {};
    folder_name_list = {};
    local_index_list = [];

    global_id = 0;

    % ------------------------------------------------------------
    % Scan each category
    % ------------------------------------------------------------
    for c = 1:size(category_defs, 1)

        folder_name = category_defs{c, 1};
        category_name = category_defs{c, 2};
        max_use_files = category_defs{c, 3};

        folder_path = fullfile(DataRoot, folder_name);

        if ~exist(folder_path, 'dir')
            warning('Folder not found: %s. Skip this category.', folder_path);
            continue;
        end

        files = list_mat_files_recursive_v1(folder_path);

        if isempty(files)
            warning('No .mat files found in: %s', folder_path);
            continue;
        end

        % Sort first, then randomize deterministically.
        [~, order] = sort({files.name});
        files = files(order);

        n_total = length(files);
        perm = randperm(n_total);

        n_use = min(max_use_files, n_total);
        selected_idx = perm(1:n_use);
        extra_idx = perm(n_use+1:end);

        % Split selected files.
        n_train = floor(train_ratio * n_use);
        n_val   = floor(val_ratio * n_use);
        n_test  = n_use - n_train - n_val;

        selected_splits = [
            repmat({'train'}, n_train, 1);
            repmat({'val'},   n_val,   1);
            repmat({'test'},  n_test,  1)
        ];

        % Append selected learning files.
        for k = 1:n_use

            f = files(selected_idx(k));

            global_id = global_id + 1;

            file_id_list{end+1,1} = sprintf('RF%06d', global_id);
            category_list{end+1,1} = category_name;
            split_list{end+1,1} = selected_splits{k};
            use_for_learning_list(end+1,1) = true;
            num_patches_list(end+1,1) = patches_per_file;

            file_path_list{end+1,1} = fullfile(f.folder, f.name);
            file_name_list{end+1,1} = f.name;
            folder_name_list{end+1,1} = folder_name;
            local_index_list(end+1,1) = k;
        end

        % Append extra files, mainly for point-target PSF holdout.
        for k = 1:length(extra_idx)

            f = files(extra_idx(k));

            global_id = global_id + 1;

            file_id_list{end+1,1} = sprintf('RF%06d', global_id);
            category_list{end+1,1} = category_name;

            if strcmp(category_name, 'simu_point')
                split_list{end+1,1} = 'extra_psf_holdout';
            else
                split_list{end+1,1} = 'extra_holdout';
            end

            use_for_learning_list(end+1,1) = false;
            num_patches_list(end+1,1) = 0;

            file_path_list{end+1,1} = fullfile(f.folder, f.name);
            file_name_list{end+1,1} = f.name;
            folder_name_list{end+1,1} = folder_name;
            local_index_list(end+1,1) = n_use + k;
        end

        fprintf('\n[%s]\n', category_name);
        fprintf('  Found files       : %d\n', n_total);
        fprintf('  Used for learning : %d\n', n_use);
        fprintf('  Train / Val / Test: %d / %d / %d\n', n_train, n_val, n_test);
        fprintf('  Extra holdout     : %d\n', length(extra_idx));
    end

    % ------------------------------------------------------------
    % Build table
    % ------------------------------------------------------------
    manifest = table( ...
        file_id_list, ...
        category_list, ...
        split_list, ...
        logical(use_for_learning_list), ...
        num_patches_list, ...
        file_path_list, ...
        file_name_list, ...
        folder_name_list, ...
        local_index_list, ...
        'VariableNames', { ...
            'file_id', ...
            'category', ...
            'split', ...
            'use_for_learning', ...
            'num_patches', ...
            'file_path', ...
            'file_name', ...
            'folder_name', ...
            'local_index'});

    % ------------------------------------------------------------
    % Save manifest
    % ------------------------------------------------------------
    csv_path = fullfile(ManifestDir, 'RF_manifest_balanced_v1.csv');
    mat_path = fullfile(ManifestDir, 'RF_manifest_balanced_v1.mat');

    writetable(manifest, csv_path);
    save(mat_path, 'manifest');

    fprintf('\n============================================================\n');
    fprintf('RF manifest saved.\n');
    fprintf('CSV: %s\n', csv_path);
    fprintf('MAT: %s\n', mat_path);
    fprintf('Total rows: %d\n', height(manifest));
    fprintf('============================================================\n');

    print_RF_manifest_summary_v1(manifest);
end

function pilot_manifest = make_pilot_manifest_v1(manifest, n_files_per_category, save_csv_path)
% ============================================================
% make_pilot_manifest_v1
%
% Purpose:
%   Create a small pilot manifest from the full manifest.
%
% It selects n_files_per_category training files from each category.
% ============================================================

    if nargin < 2 || isempty(n_files_per_category)
        n_files_per_category = 1;
    end

    if nargin < 3
        save_csv_path = '';
    end

    categories = unique(manifest.category, 'stable');

    selected_rows = false(height(manifest), 1);

    for c = 1:length(categories)

        cat_name = categories{c};

        idx = find( ...
            strcmp(manifest.category, cat_name) & ...
            strcmp(manifest.split, 'train') & ...
            manifest.use_for_learning);

        n_take = min(n_files_per_category, length(idx));

        if n_take > 0
            selected_rows(idx(1:n_take)) = true;
        end

        fprintf('[Pilot] category = %-12s, selected = %d\n', cat_name, n_take);
    end

    pilot_manifest = manifest(selected_rows, :);

    if ~isempty(save_csv_path)
        writetable(pilot_manifest, save_csv_path);
        fprintf('\nPilot manifest saved:\n%s\n', save_csv_path);
    end

    print_RF_manifest_summary_v1(pilot_manifest);
end
function print_RF_manifest_summary_v1(manifest)
% ============================================================
% print_RF_manifest_summary_v1
% ============================================================

    fprintf('\n================ Manifest Summary ================\n');

    categories = unique(manifest.category, 'stable');

    for c = 1:length(categories)

        cat_name = categories{c};

        idx_cat = strcmp(manifest.category, cat_name);

        n_total = sum(idx_cat);
        n_learning = sum(idx_cat & manifest.use_for_learning);
        n_extra = sum(idx_cat & ~manifest.use_for_learning);

        n_train = sum(idx_cat & strcmp(manifest.split, 'train'));
        n_val   = sum(idx_cat & strcmp(manifest.split, 'val'));
        n_test  = sum(idx_cat & strcmp(manifest.split, 'test'));

        fprintf('\nCategory: %s\n', cat_name);
        fprintf('  total rows       : %d\n', n_total);
        fprintf('  use_for_learning : %d\n', n_learning);
        fprintf('  train / val / test: %d / %d / %d\n', n_train, n_val, n_test);
        fprintf('  extra holdout    : %d\n', n_extra);

        n_samples = sum(manifest.num_patches(idx_cat & manifest.use_for_learning));
        fprintf('  planned samples  : %d\n', n_samples);
    end

    fprintf('\nOverall:\n');
    fprintf('  total files      : %d\n', height(manifest));
    fprintf('  learning files   : %d\n', sum(manifest.use_for_learning));
    fprintf('  planned samples  : %d\n', sum(manifest.num_patches(manifest.use_for_learning)));

    fprintf('\n==================================================\n');
end

function files = list_mat_files_recursive_v1(root_folder)
% ============================================================
% list_mat_files_recursive_v1
%
% Purpose:
%   Recursively list .mat files under root_folder.
% ============================================================

    files = dir(fullfile(root_folder, '**', '*.mat'));

    if isempty(files)
        % Fallback for older MATLAB versions.
        files = list_mat_files_recursive_fallback_v1(root_folder);
    else
        files = files(~[files.isdir]);
    end

    if ~isempty(files)
        names = {files.name};
        keep = ~startsWith(names, '.');
        files = files(keep);
    end
end

function files = list_mat_files_recursive_fallback_v1(root_folder)
% Fallback recursive .mat search.

    files = [];

    d = dir(root_folder);

    for i = 1:length(d)

        name = d(i).name;

        if strcmp(name, '.') || strcmp(name, '..')
            continue;
        end

        full_path = fullfile(root_folder, name);

        if d(i).isdir
            sub_files = list_mat_files_recursive_fallback_v1(full_path);
            files = [files; sub_files]; %#ok<AGROW>
        else
            [~, ~, ext] = fileparts(name);
            if strcmpi(ext, '.mat')
                files = [files; d(i)]; %#ok<AGROW>
                files(end).folder = root_folder;
            end
        end
    end
end