clc; clear; close all;

% ============================================================
% inspect_RF_learning_h5_meta_demo.m
%
% Purpose:
%   Inspect meta information of one RF learning sample HDF5 file.
%
% This script only reads small metadata datasets.
% It does NOT read large RF tensors such as:
%   /input/F_RC_real
%   /input/F_RC_imag
%   /input/F_CR_real
%   /input/F_CR_imag
%
% HDF5 structure assumed:
%   /sample_000001/input/...
%   /sample_000001/label/...
%   /sample_000001/baseline/...
%   /sample_000001/meta/...
% ============================================================

%% ===================== User settings =====================

save_path = 'G:\DAS\Sample\Phantom_050_learning_sample_000001.h5';

sample_group = '/sample_000001';

% Whether to print the whole HDF5 tree.
% false: only print meta summary.
% true : also run h5disp(save_path, sample_group).
show_h5_tree = false;

%% ===================== Run inspection =====================

meta = inspect_RF_learning_h5_meta_v1(save_path, sample_group, show_h5_tree);


%% ============================================================
% Local function: inspect metadata
% ============================================================

function meta = inspect_RF_learning_h5_meta_v1(save_path, sample_group, show_h5_tree)

    if nargin < 2 || isempty(sample_group)
        sample_group = '/sample_000001';
    end

    if nargin < 3 || isempty(show_h5_tree)
        show_h5_tree = false;
    end

    if sample_group(1) ~= '/'
        sample_group = ['/', sample_group];
    end

    if ~exist(save_path, 'file')
        error('File does not exist:\n%s', save_path);
    end

    fprintf('\n============================================================\n');
    fprintf('HDF5 RF learning sample inspection\n');
    fprintf('============================================================\n');
    fprintf('File         : %s\n', save_path);
    fprintf('Sample group : %s\n', sample_group);

    file_info = dir(save_path);
    fprintf('File size    : %.3f MB\n', file_info.bytes / 1024^2);

    % ------------------------------------------------------------
    % Optional: print HDF5 tree
    % ------------------------------------------------------------
    if show_h5_tree
        fprintf('\n============================================================\n');
        fprintf('HDF5 tree under %s\n', sample_group);
        fprintf('============================================================\n');
        h5disp(save_path, sample_group);
    end

    % ------------------------------------------------------------
    % Initialize output
    % ------------------------------------------------------------
    meta = struct();

    meta.save_path = save_path;
    meta.sample_group = sample_group;
    meta.file_size_MB = file_info.bytes / 1024^2;

    % ------------------------------------------------------------
    % Read sample-level attributes
    % ------------------------------------------------------------
    meta.format_version = read_h5_attr_safe(save_path, sample_group, ...
        'format_version', '');

    meta.description = read_h5_attr_safe(save_path, sample_group, ...
        'description', '');

    meta.input_tensor_order = read_h5_attr_safe(save_path, sample_group, ...
        'input_tensor_order', '');

    meta.label_order = read_h5_attr_safe(save_path, sample_group, ...
        'label_order', '');

    % ------------------------------------------------------------
    % Read meta-level attributes
    % ------------------------------------------------------------
    meta.source_file = read_h5_attr_safe(save_path, ...
        [sample_group '/meta'], 'source_file', '');

    % ------------------------------------------------------------
    % Read basic metadata datasets
    % ------------------------------------------------------------
    meta.z_idx = read_h5_vector_double(save_path, ...
        [sample_group '/meta/z_idx']);

    meta.x_idx = read_h5_vector_double(save_path, ...
        [sample_group '/meta/x_idx']);

    meta.y_idx = read_h5_vector_double(save_path, ...
        [sample_group '/meta/y_idx']);

    meta.input_angle_set = read_h5_vector_double(save_path, ...
        [sample_group '/meta/input_angle_set']);

    meta.target_angle_set = read_h5_vector_double(save_path, ...
        [sample_group '/meta/target_angle_set']);

    meta.x_axis_mm = read_h5_vector_double(save_path, ...
        [sample_group '/meta/x_axis_mm']);

    meta.y_axis_mm = read_h5_vector_double(save_path, ...
        [sample_group '/meta/y_axis_mm']);

    meta.z_axis_mm = read_h5_vector_double(save_path, ...
        [sample_group '/meta/z_axis_mm']);

    meta.frame_id = double(h5read(save_path, ...
        [sample_group '/meta/frame_id']));

    meta.patch_size = read_h5_vector_double(save_path, ...
        [sample_group '/meta/patch_size']);

    meta.input_tensor_size = read_h5_vector_double(save_path, ...
        [sample_group '/meta/input_tensor_size']);

    meta.label_size = read_h5_vector_double(save_path, ...
        [sample_group '/meta/label_size']);

    % ------------------------------------------------------------
    % Optional debug metadata
    % ------------------------------------------------------------
    path_RC_pos = [sample_group '/meta/input_sample_pos_RC_minmax'];
    path_CR_pos = [sample_group '/meta/input_sample_pos_CR_minmax'];
    path_RC_apo = [sample_group '/meta/input_apo_count_RC'];
    path_CR_apo = [sample_group '/meta/input_apo_count_CR'];

    if has_h5_dataset_local(save_path, path_RC_pos)
        meta.input_sample_pos_RC_minmax = double(h5read(save_path, path_RC_pos));
    end

    if has_h5_dataset_local(save_path, path_CR_pos)
        meta.input_sample_pos_CR_minmax = double(h5read(save_path, path_CR_pos));
    end

    if has_h5_dataset_local(save_path, path_RC_apo)
        meta.input_apo_count_RC = read_h5_vector_double(save_path, path_RC_apo);
    end

    if has_h5_dataset_local(save_path, path_CR_apo)
        meta.input_apo_count_CR = read_h5_vector_double(save_path, path_CR_apo);
    end

    % ------------------------------------------------------------
    % Dataset size inspection without reading large tensors
    % ------------------------------------------------------------
    meta.dataset_info = inspect_main_dataset_sizes_v1(save_path, sample_group);

    % ------------------------------------------------------------
    % Print summary
    % ------------------------------------------------------------
    print_RF_learning_meta_summary_v1(meta);

end


%% ============================================================
% Local function: print summary
% ============================================================

function print_RF_learning_meta_summary_v1(meta)

    fprintf('\n============================================================\n');
    fprintf('Basic information\n');
    fprintf('============================================================\n');

    fprintf('Format version : %s\n', string(meta.format_version));
    fprintf('Source file    : %s\n', string(meta.source_file));
    fprintf('Frame ID       : %d\n', meta.frame_id);

    if ~isempty(meta.description)
        fprintf('Description    : %s\n', string(meta.description));
    end

    fprintf('\n============================================================\n');
    fprintf('Patch index information\n');
    fprintf('============================================================\n');

    print_index_range('z_idx', meta.z_idx);
    print_index_range('x_idx', meta.x_idx);
    print_index_range('y_idx', meta.y_idx);

    fprintf('\nPatch size from meta: [%s]\n', num2str(meta.patch_size));

    fprintf('\n============================================================\n');
    fprintf('Physical coordinate information [mm]\n');
    fprintf('============================================================\n');

    print_axis_range('z_axis_mm', meta.z_axis_mm);
    print_axis_range('x_axis_mm', meta.x_axis_mm);
    print_axis_range('y_axis_mm', meta.y_axis_mm);

    fprintf('\n============================================================\n');
    fprintf('Angle information\n');
    fprintf('============================================================\n');

    fprintf('Input angle set  (%d angles):\n', numel(meta.input_angle_set));
    disp(meta.input_angle_set);

    fprintf('Target angle set (%d angles):\n', numel(meta.target_angle_set));
    disp(meta.target_angle_set);

    fprintf('\n============================================================\n');
    fprintf('Tensor size information\n');
    fprintf('============================================================\n');

    fprintf('Input tensor size from meta : [%s]\n', ...
        num2str(meta.input_tensor_size));

    fprintf('Label size from meta        : [%s]\n', ...
        num2str(meta.label_size));

    fprintf('Input tensor order          : %s\n', ...
        string(meta.input_tensor_order));

    fprintf('Label order                 : %s\n', ...
        string(meta.label_order));

    fprintf('\n============================================================\n');
    fprintf('Actual HDF5 dataset sizes without loading large tensors\n');
    fprintf('============================================================\n');

    names = fieldnames(meta.dataset_info);

    for i = 1:numel(names)
        item = meta.dataset_info.(names{i});

        if item.exists
            fprintf('%-35s : [%s], %s\n', ...
                item.path, num2str(item.size), item.datatype);
        else
            fprintf('%-35s : not found\n', item.path);
        end
    end

    if isfield(meta, 'input_sample_pos_RC_minmax') || ...
       isfield(meta, 'input_sample_pos_CR_minmax')

        fprintf('\n============================================================\n');
        fprintf('Debug: RF sample position range\n');
        fprintf('============================================================\n');

        if isfield(meta, 'input_sample_pos_RC_minmax')
            fprintf('RC sample position min/max per input angle:\n');
            disp(meta.input_sample_pos_RC_minmax);
        end

        if isfield(meta, 'input_sample_pos_CR_minmax')
            fprintf('CR sample position min/max per input angle:\n');
            disp(meta.input_sample_pos_CR_minmax);
        end
    end

    if isfield(meta, 'input_apo_count_RC') || ...
       isfield(meta, 'input_apo_count_CR')

        fprintf('\n============================================================\n');
        fprintf('Debug: receive apodization count\n');
        fprintf('============================================================\n');

        if isfield(meta, 'input_apo_count_RC')
            fprintf('RC mean valid apodization count per angle:\n');
            disp(meta.input_apo_count_RC);
        end

        if isfield(meta, 'input_apo_count_CR')
            fprintf('CR mean valid apodization count per angle:\n');
            disp(meta.input_apo_count_CR);
        end
    end

    fprintf('\n============================================================\n');
    fprintf('Inspection finished.\n');
    fprintf('============================================================\n\n');

end


%% ============================================================
% Local function: inspect main dataset sizes
% ============================================================

function dataset_info = inspect_main_dataset_sizes_v1(save_path, sample_group)

    dataset_info = struct();

    dataset_list = { ...
        'input_F_RC_real',        [sample_group '/input/F_RC_real']; ...
        'input_F_RC_imag',        [sample_group '/input/F_RC_imag']; ...
        'input_F_CR_real',        [sample_group '/input/F_CR_real']; ...
        'input_F_CR_imag',        [sample_group '/input/F_CR_imag']; ...
        'label_DAS_target_real',  [sample_group '/label/DAS_target_real']; ...
        'label_DAS_target_imag',  [sample_group '/label/DAS_target_imag']; ...
        'label_DAS_target_abs',   [sample_group '/label/DAS_target_abs']; ...
        'baseline_DAS_input_real',[sample_group '/baseline/DAS_input_real']; ...
        'baseline_DAS_input_imag',[sample_group '/baseline/DAS_input_imag']; ...
        'baseline_DAS_input_abs', [sample_group '/baseline/DAS_input_abs'] ...
        };

    for i = 1:size(dataset_list, 1)

        field_name = dataset_list{i, 1};
        dataset_path = dataset_list{i, 2};

        item = struct();
        item.path = dataset_path;
        item.exists = false;
        item.size = [];
        item.datatype = '';

        try
            info = h5info(save_path, dataset_path);

            item.exists = true;
            item.size = info.Dataspace.Size;
            item.datatype = info.Datatype.Class;

        catch
            item.exists = false;
        end

        dataset_info.(field_name) = item;
    end

end


%% ============================================================
% Local helper functions
% ============================================================

function v = read_h5_vector_double(save_path, dataset_path)

    v = double(h5read(save_path, dataset_path));
    v = v(:).';

end


function attr_value = read_h5_attr_safe(save_path, object_path, attr_name, default_value)

    try
        attr_value = h5readatt(save_path, object_path, attr_name);
    catch
        attr_value = default_value;
    end

end


function tf = has_h5_dataset_local(save_path, dataset_path)

    tf = false;

    try
        h5info(save_path, dataset_path);
        tf = true;
    catch
        tf = false;
    end

end


function print_index_range(name, idx)

    if isempty(idx)
        fprintf('%s: empty\n', name);
        return;
    end

    fprintf('%s: %d ~ %d, count = %d\n', ...
        name, idx(1), idx(end), numel(idx));

end


function print_axis_range(name, axis_mm)

    if isempty(axis_mm)
        fprintf('%s: empty\n', name);
        return;
    end

    if numel(axis_mm) >= 2
        spacing = mean(diff(axis_mm));
    else
        spacing = NaN;
    end

    fprintf('%s: %.6f ~ %.6f mm, count = %d, mean spacing = %.6f mm\n', ...
        name, axis_mm(1), axis_mm(end), numel(axis_mm), spacing);

end