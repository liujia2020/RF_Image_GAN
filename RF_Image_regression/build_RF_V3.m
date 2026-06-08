clc; clear; close all;
% ============================================================
% Dense tiled RF learning sample generation
%
% Purpose:
%   Generate one carotid test volume as non-overlapping dense patches
%   for sliding-window inference / reconstruction checks.
%
% Dense target:
%   Full scan size : [1024, 128, 128] = [Nz, Nx, Ny]
%   Patch size     : [32, 16, 16]
%   Stride         : patch size, no overlap
%   Patch count    : 32 * 8 * 8 = 2048
%
% Important convention:
%   HDF5 meta z_idx/x_idx/y_idx are MATLAB 1-based indices, matching
%   the existing sparse samples. Python reconstruction must subtract 1
%   before using these values as array indices.
% ============================================================

% Safety switch. Set true only when you are ready to generate HDF5 files.
run_generation = true;

if ~run_generation
    fprintf('Dense carotid test generation is disabled. Set run_generation = true to run.\n');
    return;
end

% ------------------------------------------------------------
% Paths
% ------------------------------------------------------------
DataRoot = 'F:\Data_0110_RFdata';
ManifestDir = fullfile(DataRoot, '_manifest');
FullManifestCsv = fullfile(ManifestDir, ...
    'RF_manifest_filelevel_full_70train_15val_15test.csv');

OutputRoot = 'F:\DAS\RF_LearningSamples_dense_carotid_test_32x16x16';
DenseManifestCsv = fullfile(ManifestDir, ...
    'RF_manifest_dense_carotid_test_32x16x16_onefile.csv');

% ------------------------------------------------------------
% Sample settings
% ------------------------------------------------------------
input_angle_set  = [3, 38, 73];
target_angle_set = round(linspace(1, 75, 33));
patch_size = [32, 16, 16];

% Empty means: use the first carotid test RF file in the manifest.
% To force one file, set for example: dense_test_file_id = 'RF000486';
dense_test_file_id = '';

% ------------------------------------------------------------
% Select one carotid test volume
% ------------------------------------------------------------
manifest = readtable(FullManifestCsv);

keep = ...
    strcmp(string(manifest.category), 'carotid') & ...
    strcmp(string(manifest.split), 'test') & ...
    logical(manifest.use_for_learning);

if ~isempty(dense_test_file_id)
    keep = keep & strcmp(string(manifest.file_id), dense_test_file_id);
end

dense_manifest = manifest(keep, :);

if isempty(dense_manifest)
    error('No carotid test RF file found for dense tiled generation.');
end

dense_manifest = dense_manifest(1, :);
writetable(dense_manifest, DenseManifestCsv);

selected_file_id = get_table_value_as_char_v1(dense_manifest.file_id(1));
selected_category = get_table_value_as_char_v1(dense_manifest.category(1));
selected_split = get_table_value_as_char_v1(dense_manifest.split(1));
selected_file_path = get_table_value_as_char_v1(dense_manifest.file_path(1));

fprintf('\nSelected dense test volume:\n');
fprintf('  file_id   : %s\n', selected_file_id);
fprintf('  category  : %s\n', selected_category);
fprintf('  split     : %s\n', selected_split);
fprintf('  file_path : %s\n', selected_file_path);
fprintf('  manifest  : %s\n', DenseManifestCsv);
fprintf('  output    : %s\n', OutputRoot);

% ------------------------------------------------------------
% Generation options
% ------------------------------------------------------------
gen_opts = struct();
gen_opts.overwrite = false;
gen_opts.validate_after_save = false;
gen_opts.patch_grid_mode = 'dense';
gen_opts.max_files = 1;

% ------------------------------------------------------------
% Generate dense HDF5 patches
% ------------------------------------------------------------
log_table = generate_RF_learning_samples_from_manifest_v1( ...
    DenseManifestCsv, OutputRoot, ...
    input_angle_set, target_angle_set, ...
    patch_size, gen_opts);

% ------------------------------------------------------------
% Quick post-generation check
% ------------------------------------------------------------
save_dir = fullfile(OutputRoot, selected_split, selected_category);
file_pattern = sprintf('%s_%s_%s_patch*.h5', ...
    selected_file_id, selected_category, selected_split);
dense_files = dir(fullfile(save_dir, file_pattern));

fprintf('\nDense output check:\n');
fprintf('  output directory : %s\n', save_dir);
fprintf('  matching files   : %d\n', numel(dense_files));
fprintf('  expected files   : 2048\n');

if ~isempty(dense_files)
    probe_path = fullfile(save_dir, dense_files(1).name);

    z = h5read(probe_path, '/sample_000001/meta/z_idx');
    x = h5read(probe_path, '/sample_000001/meta/x_idx');
    y = h5read(probe_path, '/sample_000001/meta/y_idx');
    frame_id = h5read(probe_path, '/sample_000001/meta/frame_id');
    source_file = h5readatt(probe_path, '/sample_000001/meta', 'source_file');

    fprintf('\nProbe dense patch meta:\n');
    fprintf('  file        : %s\n', probe_path);
    fprintf('  z_idx       : first=%d last=%d min=%d max=%d len=%d\n', ...
        z(1), z(end), min(z), max(z), numel(z));
    fprintf('  x_idx       : first=%d last=%d min=%d max=%d len=%d\n', ...
        x(1), x(end), min(x), max(x), numel(x));
    fprintf('  y_idx       : first=%d last=%d min=%d max=%d len=%d\n', ...
        y(1), y(end), min(y), max(y), numel(y));
    fprintf('  source_file : %s\n', source_file);
    fprintf('  frame_id    : %d\n', frame_id);
end

%% Local functions copied from build_RF_datase_V2.m
function out_path = h5path(base_path, sub_path)
% Join HDF5 paths.

    if base_path(end) == '/'
        base_path = base_path(1:end-1);
    end

    if sub_path(1) == '/'
        sub_path = sub_path(2:end);
    end

    out_path = [base_path, '/', sub_path];
end
function write_h5_single_dataset(save_path, dataset_name, data)
% Write a single-precision HDF5 dataset.

    data = single(data);

    h5create(save_path, dataset_name, size(data), ...
        'Datatype', 'single');

    h5write(save_path, dataset_name, data);
end
function write_h5_int32_dataset(save_path, dataset_name, data)
% Write an int32 HDF5 dataset.

    data = int32(data);

    h5create(save_path, dataset_name, size(data), ...
        'Datatype', 'int32');

    h5write(save_path, dataset_name, data);
end
function tf = has_h5_dataset(file_path, dataset_name)
% Return true if a dataset exists in an HDF5 file.
%
% dataset_name example:
%   '/F_RC_real'

    tf = false;

    try
        info = h5info(file_path);
        tf = search_h5_dataset_recursive(info, dataset_name);
    catch
        tf = false;
    end
end

function out = extract_delay_aligned_RF_patch_v1(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                                 angle_set, z_idx, x_idx, y_idx)
% ============================================================
% extract_delay_aligned_RF_patch_v1
%
% Purpose:
%   Extract delay-aligned complex RF patch tensors for RCA OPW DAS.
%
% Inputs:
%   RcvData     : raw RF data for one frame, typically [Nt, 256]
%   Trans       : Verasonics Trans structure
%   Resource    : Verasonics Resource structure
%   TX          : Verasonics TX structure
%   TW          : Verasonics TW structure
%   Receive     : Receive structure, one entry per TX event
%   scan        : full DAS scan grid
%   angle_set   : selected local angle indices, e.g. [3, 38, 73]
%   z_idx       : global z indices of the patch
%   x_idx       : global x indices of the patch
%   y_idx       : global y indices of the patch
%
% Outputs:
%   out.F_RC_patch : [Nz, Nx, Ny, 128, N_angle], complex single
%   out.F_CR_patch : [Nz, Nx, Ny, 128, N_angle], complex single
%
% Notes:
%   - This function only extracts RF-derived tensors.
%   - It does not validate, plot, save, normalize, envelope-detect,
%     convert to dB, or split real/imag.
%   - It assumes extract_rf_subset(...) already exists in your script.
% ============================================================

    % ------------------------------------------------------------
    % 1. Build patch scan grid from the original scan
    % ------------------------------------------------------------
    scan_patch = struct();

    scan_patch.x_axis = scan.x_axis(x_idx);
    scan_patch.y_axis = scan.y_axis(y_idx);
    scan_patch.z_axis = scan.z_axis(z_idx);

    scan_patch.N_z = length(z_idx);
    scan_patch.N_x = length(x_idx);
    scan_patch.N_y = length(y_idx);

    [scan_patch.x, scan_patch.z, scan_patch.y] = meshgrid( ...
        scan_patch.x_axis, scan_patch.z_axis, scan_patch.y_axis);

    scan_patch.N_pixels = numel(scan_patch.x);

    Nz = scan_patch.N_z;
    Nx = scan_patch.N_x;
    Ny = scan_patch.N_y;
    Nvox = scan_patch.N_pixels;

    N_ch = 128;
    N_angle = length(angle_set);

    % ------------------------------------------------------------
    % 2. Basic parameters
    % ------------------------------------------------------------
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;

    c0 = 1540;
    lambda = c0 / f0;

    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;
    rx_f_number = 1.5;
    offset_distance = TW.peak * lambda;

    % ------------------------------------------------------------
    % 3. Extract RC / CR RF data and apply Hilbert transform
    % ------------------------------------------------------------
    RF_RC = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                              'RC', angle_set, half_waves);
    RF_RC = hilbert(RF_RC);

    RF_CR = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                              'CR', angle_set, half_waves);
    RF_CR = hilbert(RF_CR);

    time_vector = (0:(size(RF_RC, 1)-1)) / fs;

    % ------------------------------------------------------------
    % 4. Allocate output tensors
    % ------------------------------------------------------------
    F_RC_patch = complex(zeros(Nz, Nx, Ny, N_ch, N_angle, 'single'));
    F_CR_patch = complex(zeros(Nz, Nx, Ny, N_ch, N_angle, 'single'));

    sample_pos_RC_minmax = zeros(N_angle, 2);
    sample_pos_CR_minmax = zeros(N_angle, 2);

    apo_count_RC = zeros(N_angle, 1);
    apo_count_CR = zeros(N_angle, 1);

    % ============================================================
    % 5. RC branch
    % ============================================================

    % RC angle list: first half TX, Steer(1)
    alpha = zeros(half_waves, 1);
    for k = 1:half_waves
        alpha(k) = TX(k).Steer(1);
    end

    % RC receive aperture: ElementPos(129:256), y-z receive plane
    CRh_probe.x = ElementPos(129:256, 1);
    CRh_probe.y = ElementPos(129:256, 2);
    CRh_probe.z = ElementPos(129:256, 3);

    Cym = CRh_probe.y.' - scan_patch.y(:);
    Czm = CRh_probe.z.' - scan_patch.z(:);

    receive_delay_RC = single(sqrt(Cym.^2 + Czm.^2) / c0);
    apo_RC = single(abs(rx_f_number .* Cym ./ Czm) <= 0.5);

    D_RC = abs(CRh_probe.y(end) - CRh_probe.y(1));

    fprintf('\n=== extract_delay_aligned_RF_patch_v1: RC branch ===\n');

    for ia = 1:N_angle

        n_wave = angle_set(ia);
        angle_val = alpha(n_wave);

        transmit_distance = scan_patch.z(:) * cos(angle_val) + ...
                            scan_patch.x(:) * sin(angle_val) + ...
                            (D_RC/2) * sin(angle_val) * sign(angle_val) + ...
                            offset_distance;

        transmit_time = transmit_distance ./ c0;

        all_sample_pos = zeros(Nvox, N_ch);

        for n_rx = 1:N_ch

            delay = receive_delay_RC(:, n_rx) + single(transmit_time);

            sample_pos = double(delay) * fs + 1;
            all_sample_pos(:, n_rx) = sample_pos;

            temp = apo_RC(:, n_rx) .* ...
                   interp1(time_vector, RF_RC(:, n_rx, ia), ...
                           double(delay), 'linear', 0);

            F_RC_patch(:, :, :, n_rx, ia) = reshape(temp, [Nz, Nx, Ny]);
        end

        sample_pos_RC_minmax(ia, :) = [min(all_sample_pos(:)), max(all_sample_pos(:))];
        apo_count_RC(ia) = sum(apo_RC(:) == 1) / Nvox;

        fprintf('[RC %d/%d] angle = %.3f deg, sample pos = %.2f ~ %.2f, mean valid apo = %.1f / 128\n', ...
            ia, N_angle, angle_val * 180/pi, ...
            sample_pos_RC_minmax(ia, 1), sample_pos_RC_minmax(ia, 2), ...
            apo_count_RC(ia));
    end

    % ============================================================
    % 6. CR branch
    % ============================================================

    % CR angle list: second half TX, Steer(2)
    beta = zeros(half_waves, 1);
    for k = 1:half_waves
        beta(k) = TX(half_waves + k).Steer(2);
    end

    % CR receive aperture: ElementPos(1:128), x-z receive plane
    RRh_probe.x = ElementPos(1:128, 1);
    RRh_probe.y = ElementPos(1:128, 2);
    RRh_probe.z = ElementPos(1:128, 3);

    Rxm = RRh_probe.x.' - scan_patch.x(:);
    Rzm = RRh_probe.z.' - scan_patch.z(:);

    receive_delay_CR = single(sqrt(Rxm.^2 + Rzm.^2) / c0);
    apo_CR = single(abs(rx_f_number .* Rxm ./ Rzm) <= 0.5);

    D_CR = abs(RRh_probe.x(end) - RRh_probe.x(1));

    fprintf('\n=== extract_delay_aligned_RF_patch_v1: CR branch ===\n');

    for ia = 1:N_angle

        n_wave = angle_set(ia);
        angle_val = beta(n_wave);

        transmit_distance = scan_patch.z(:) * cos(angle_val) + ...
                            scan_patch.y(:) * sin(angle_val) + ...
                            (D_CR/2) * sin(angle_val) * sign(angle_val) + ...
                            offset_distance;

        transmit_time = transmit_distance ./ c0;

        all_sample_pos = zeros(Nvox, N_ch);

        for n_rx = 1:N_ch

            delay = receive_delay_CR(:, n_rx) + single(transmit_time);

            sample_pos = double(delay) * fs + 1;
            all_sample_pos(:, n_rx) = sample_pos;

            temp = apo_CR(:, n_rx) .* ...
                   interp1(time_vector, RF_CR(:, n_rx, ia), ...
                           double(delay), 'linear', 0);

            F_CR_patch(:, :, :, n_rx, ia) = reshape(temp, [Nz, Nx, Ny]);
        end

        sample_pos_CR_minmax(ia, :) = [min(all_sample_pos(:)), max(all_sample_pos(:))];
        apo_count_CR(ia) = sum(apo_CR(:) == 1) / Nvox;

        fprintf('[CR %d/%d] angle = %.3f deg, sample pos = %.2f ~ %.2f, mean valid apo = %.1f / 128\n', ...
            ia, N_angle, angle_val * 180/pi, ...
            sample_pos_CR_minmax(ia, 1), sample_pos_CR_minmax(ia, 2), ...
            apo_count_CR(ia));
    end

    % ------------------------------------------------------------
    % 7. Output
    % ------------------------------------------------------------
    out = struct();

    out.F_RC_patch = F_RC_patch;
    out.F_CR_patch = F_CR_patch;

    out.scan_patch = scan_patch;

    out.z_idx = z_idx;
    out.x_idx = x_idx;
    out.y_idx = y_idx;
    out.angle_set = angle_set;

    out.sample_pos_RC_minmax = sample_pos_RC_minmax;
    out.sample_pos_CR_minmax = sample_pos_CR_minmax;

    out.apo_count_RC = apo_count_RC;
    out.apo_count_CR = apo_count_CR;

    out.fs = fs;
    out.f0 = f0;
    out.c0 = c0;
    out.lambda = lambda;

    fprintf('\n=== RF patch extraction finished ===\n');
    fprintf('F_RC_patch size: [%s]\n', num2str(size(F_RC_patch)));
    fprintf('F_CR_patch size: [%s]\n', num2str(size(F_CR_patch)));
end
function rf = extract_rf_subset(RcvData, Receive, Resource, Trans, mode, angle_indices, half_waves)
    endSample = Receive(1).endSample;
    total_waves = half_waves * 2; 
    
    if strcmp(mode, 'CR')
        wave_offset = half_waves; elem_indices = 1:128;
    else
        wave_offset = 0; elem_indices = 129:256;
    end
    
    adc_channels = Trans.Connector(elem_indices);
    
    num_angles = length(angle_indices);
    rf = zeros(endSample, length(elem_indices), num_angles, 'single');
    f = 1; 
    
    if iscell(RcvData), src = RcvData{1}; else, src = RcvData; end
    
    for i = 1:num_angles
        w_local = angle_indices(i); 
        w_global = wave_offset + w_local; 
        idx = (f-1)*total_waves + w_global;
        rf(:,:,i) = single(src(Receive(idx).startSample:Receive(idx).endSample, adc_channels, f));
    end
end

function save_RF_learning_sample_h5_v2(sample, save_path, sample_group)
% ============================================================
% save_RF_learning_sample_h5_v2
%
% Save one paired learning sample to HDF5.
% ============================================================

    if nargin < 3 || isempty(sample_group)
        sample_group = '/sample_000001';
    end

    if sample_group(1) ~= '/'
        sample_group = ['/', sample_group];
    end

    if exist(save_path, 'file')
        delete(save_path);
    end

    F_RC = single(sample.input.F_RC_patch);
    F_CR = single(sample.input.F_CR_patch);

    % ------------------------------------------------------------
    % 1. Input RF tensors
    % ------------------------------------------------------------
    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_RC_real'), single(real(F_RC)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_RC_imag'), single(imag(F_RC)));

    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_CR_real'), single(real(F_CR)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_CR_imag'), single(imag(F_CR)));

    % ------------------------------------------------------------
    % 2. Baseline: input-angle DAS sum and avg
    % ------------------------------------------------------------
    DAS_input_sum = single(sample.baseline.DAS_input_sum);
    DAS_input_avg = single(sample.baseline.DAS_input_avg);

    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_sum_real'), single(real(DAS_input_sum)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_sum_imag'), single(imag(DAS_input_sum)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_sum_abs'),  single(sample.baseline.DAS_input_sum_abs));

    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_avg_real'), single(real(DAS_input_avg)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_avg_imag'), single(imag(DAS_input_avg)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_avg_abs'),  single(sample.baseline.DAS_input_avg_abs));

    % ------------------------------------------------------------
    % 3. Label: target-angle DAS sum and avg
    % ------------------------------------------------------------
    DAS_target_sum = single(sample.label.DAS_target_sum);
    DAS_target_avg = single(sample.label.DAS_target_avg);

    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_sum_real'), single(real(DAS_target_sum)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_sum_imag'), single(imag(DAS_target_sum)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_sum_abs'),  single(sample.label.DAS_target_sum_abs));

    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_avg_real'), single(real(DAS_target_avg)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_avg_imag'), single(imag(DAS_target_avg)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_avg_abs'),  single(sample.label.DAS_target_avg_abs));

    % ------------------------------------------------------------
    % 4. Metadata
    % ------------------------------------------------------------
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/z_idx'), int32(sample.meta.z_idx(:)));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/x_idx'), int32(sample.meta.x_idx(:)));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/y_idx'), int32(sample.meta.y_idx(:)));

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/input_angle_set'), int32(sample.meta.input_angle_set(:)));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/target_angle_set'), int32(sample.meta.target_angle_set(:)));

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/input_angle_count'), int32(sample.meta.input_angle_count));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/target_angle_count'), int32(sample.meta.target_angle_count));

    write_h5_single_dataset(save_path, h5path(sample_group, 'meta/x_axis_mm'), single(sample.meta.x_axis_mm(:)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'meta/y_axis_mm'), single(sample.meta.y_axis_mm(:)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'meta/z_axis_mm'), single(sample.meta.z_axis_mm(:)));

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/frame_id'), int32(sample.meta.frame_id));

    patch_size = int32([ ...
        sample.meta.scan_patch.N_z; ...
        sample.meta.scan_patch.N_x; ...
        sample.meta.scan_patch.N_y]);

    input_tensor_size = int32(size(F_RC)).';
    label_size = int32(size(DAS_target_avg)).';

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/patch_size'), patch_size);
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/input_tensor_size'), input_tensor_size);
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/label_size'), label_size);

    % Optional input debug metadata
    if isfield(sample.meta, 'input_sample_pos_RC_minmax')
        write_h5_single_dataset(save_path, h5path(sample_group, 'meta/input_sample_pos_RC_minmax'), ...
            single(sample.meta.input_sample_pos_RC_minmax));
    end

    if isfield(sample.meta, 'input_sample_pos_CR_minmax')
        write_h5_single_dataset(save_path, h5path(sample_group, 'meta/input_sample_pos_CR_minmax'), ...
            single(sample.meta.input_sample_pos_CR_minmax));
    end

    if isfield(sample.meta, 'input_apo_count_RC')
        write_h5_single_dataset(save_path, h5path(sample_group, 'meta/input_apo_count_RC'), ...
            single(sample.meta.input_apo_count_RC));
    end

    if isfield(sample.meta, 'input_apo_count_CR')
        write_h5_single_dataset(save_path, h5path(sample_group, 'meta/input_apo_count_CR'), ...
            single(sample.meta.input_apo_count_CR));
    end

    % Attributes
    try
        h5writeatt(save_path, sample_group, 'format_version', 'RF_learning_sample_h5_v2');
        h5writeatt(save_path, sample_group, 'description', ...
            'Paired sample: few-angle delay-aligned RF tensor input and target-angle DAS label.');
        h5writeatt(save_path, sample_group, 'input_tensor_order', ...
            '[Nz, Nx, Ny, N_channel, N_angle]');
        h5writeatt(save_path, sample_group, 'label_order', ...
            '[Nz, Nx, Ny]');
        h5writeatt(save_path, sample_group, 'recommended_training_label', ...
            'label/DAS_target_avg_real + label/DAS_target_avg_imag');
        h5writeatt(save_path, h5path(sample_group, 'meta'), 'source_file', sample.meta.source_file);
    catch
        warning('Could not write one or more HDF5 attributes.');
    end

    file_info = dir(save_path);

    fprintf('\n=== save_RF_learning_sample_h5_v2 finished ===\n');
    fprintf('Saved file   : %s\n', save_path);
    fprintf('Sample group : %s\n', sample_group);
    fprintf('Input F_RC   : [%s]\n', num2str(size(F_RC)));
    fprintf('Input F_CR   : [%s]\n', num2str(size(F_CR)));
    fprintf('Label avg    : [%s]\n', num2str(size(DAS_target_avg)));
    fprintf('Baseline avg : [%s]\n', num2str(size(DAS_input_avg)));
    fprintf('File size    : %.3f MB\n', file_info.bytes / 1024^2);
end
function val = validate_RF_learning_sample_h5_v2(sample, loaded)
% ============================================================
% validate_RF_learning_sample_h5_v2
% ============================================================

    val = struct();

    % ------------------------------------------------------------
    % 1. Input tensor
    % ------------------------------------------------------------
    diff_RC = single(sample.input.F_RC_patch) - single(loaded.input.F_RC_patch);
    diff_CR = single(sample.input.F_CR_patch) - single(loaded.input.F_CR_patch);

    val.input_RC_max_abs_diff = max(abs(diff_RC(:)));
    val.input_CR_max_abs_diff = max(abs(diff_CR(:)));

    val.input_RC_rel_l2 = norm(diff_RC(:)) / ...
        (norm(single(sample.input.F_RC_patch(:))) + eps);

    val.input_CR_rel_l2 = norm(diff_CR(:)) / ...
        (norm(single(sample.input.F_CR_patch(:))) + eps);

    % ------------------------------------------------------------
    % 2. Baseline sum / avg
    % ------------------------------------------------------------
    diff_baseline_sum = single(sample.baseline.DAS_input_sum) - single(loaded.baseline.DAS_input_sum);
    diff_baseline_avg = single(sample.baseline.DAS_input_avg) - single(loaded.baseline.DAS_input_avg);

    val.baseline_sum_max_abs_diff = max(abs(diff_baseline_sum(:)));
    val.baseline_avg_max_abs_diff = max(abs(diff_baseline_avg(:)));

    val.baseline_sum_rel_l2 = norm(diff_baseline_sum(:)) / ...
        (norm(single(sample.baseline.DAS_input_sum(:))) + eps);

    val.baseline_avg_rel_l2 = norm(diff_baseline_avg(:)) / ...
        (norm(single(sample.baseline.DAS_input_avg(:))) + eps);

    % ------------------------------------------------------------
    % 3. Label sum / avg
    % ------------------------------------------------------------
    diff_label_sum = single(sample.label.DAS_target_sum) - single(loaded.label.DAS_target_sum);
    diff_label_avg = single(sample.label.DAS_target_avg) - single(loaded.label.DAS_target_avg);

    val.label_sum_max_abs_diff = max(abs(diff_label_sum(:)));
    val.label_avg_max_abs_diff = max(abs(diff_label_avg(:)));

    val.label_sum_rel_l2 = norm(diff_label_sum(:)) / ...
        (norm(single(sample.label.DAS_target_sum(:))) + eps);

    val.label_avg_rel_l2 = norm(diff_label_avg(:)) / ...
        (norm(single(sample.label.DAS_target_avg(:))) + eps);

    % ------------------------------------------------------------
    % 4. Recompute baseline from loaded input tensor
    % ------------------------------------------------------------
    DAS_input_sum_from_loaded_RF = sum(sum(loaded.input.F_RC_patch, 5), 4) + ...
                                   sum(sum(loaded.input.F_CR_patch, 5), 4);

    DAS_input_sum_from_loaded_RF = reshape(single(DAS_input_sum_from_loaded_RF), ...
        size(loaded.baseline.DAS_input_sum));

    DAS_input_avg_from_loaded_RF = DAS_input_sum_from_loaded_RF ./ ...
        single(loaded.meta.input_angle_count);

    diff_recomputed_sum = DAS_input_sum_from_loaded_RF - single(loaded.baseline.DAS_input_sum);
    diff_recomputed_avg = DAS_input_avg_from_loaded_RF - single(loaded.baseline.DAS_input_avg);

    val.loaded_RF_vs_baseline_sum_max_abs_diff = max(abs(diff_recomputed_sum(:)));
    val.loaded_RF_vs_baseline_avg_max_abs_diff = max(abs(diff_recomputed_avg(:)));

    val.loaded_RF_vs_baseline_sum_rel_l2 = norm(diff_recomputed_sum(:)) / ...
        (norm(single(loaded.baseline.DAS_input_sum(:))) + eps);

    val.loaded_RF_vs_baseline_avg_rel_l2 = norm(diff_recomputed_avg(:)) / ...
        (norm(single(loaded.baseline.DAS_input_avg(:))) + eps);

    % ------------------------------------------------------------
    % 5. Metadata
    % ------------------------------------------------------------
    val.same_z_idx = isequal(double(sample.meta.z_idx(:)).', double(loaded.meta.z_idx(:)).');
    val.same_x_idx = isequal(double(sample.meta.x_idx(:)).', double(loaded.meta.x_idx(:)).');
    val.same_y_idx = isequal(double(sample.meta.y_idx(:)).', double(loaded.meta.y_idx(:)).');

    val.same_input_angle_set = isequal(double(sample.meta.input_angle_set(:)).', ...
                                       double(loaded.meta.input_angle_set(:)).');

    val.same_target_angle_set = isequal(double(sample.meta.target_angle_set(:)).', ...
                                        double(loaded.meta.target_angle_set(:)).');

    % ------------------------------------------------------------
    % 6. Print
    % ------------------------------------------------------------
    fprintf('\n=== validate_RF_learning_sample_h5_v2 ===\n');

    fprintf('\n[Input RF tensor]\n');
    fprintf('F_RC max abs diff = %.6e\n', val.input_RC_max_abs_diff);
    fprintf('F_RC rel L2       = %.6e\n', val.input_RC_rel_l2);
    fprintf('F_CR max abs diff = %.6e\n', val.input_CR_max_abs_diff);
    fprintf('F_CR rel L2       = %.6e\n', val.input_CR_rel_l2);

    fprintf('\n[Baseline DAS]\n');
    fprintf('DAS input sum max abs diff = %.6e\n', val.baseline_sum_max_abs_diff);
    fprintf('DAS input sum rel L2       = %.6e\n', val.baseline_sum_rel_l2);
    fprintf('DAS input avg max abs diff = %.6e\n', val.baseline_avg_max_abs_diff);
    fprintf('DAS input avg rel L2       = %.6e\n', val.baseline_avg_rel_l2);

    fprintf('\n[Label DAS]\n');
    fprintf('DAS target sum max abs diff = %.6e\n', val.label_sum_max_abs_diff);
    fprintf('DAS target sum rel L2       = %.6e\n', val.label_sum_rel_l2);
    fprintf('DAS target avg max abs diff = %.6e\n', val.label_avg_max_abs_diff);
    fprintf('DAS target avg rel L2       = %.6e\n', val.label_avg_rel_l2);

    fprintf('\n[Loaded input RF -> baseline check]\n');
    fprintf('sum max abs diff = %.6e\n', val.loaded_RF_vs_baseline_sum_max_abs_diff);
    fprintf('sum rel L2       = %.6e\n', val.loaded_RF_vs_baseline_sum_rel_l2);
    fprintf('avg max abs diff = %.6e\n', val.loaded_RF_vs_baseline_avg_max_abs_diff);
    fprintf('avg rel L2       = %.6e\n', val.loaded_RF_vs_baseline_avg_rel_l2);

    fprintf('\n[Metadata]\n');
    fprintf('same z_idx             = %d\n', val.same_z_idx);
    fprintf('same x_idx             = %d\n', val.same_x_idx);
    fprintf('same y_idx             = %d\n', val.same_y_idx);
    fprintf('same input_angle_set   = %d\n', val.same_input_angle_set);
    fprintf('same target_angle_set  = %d\n', val.same_target_angle_set);
end

function value = get_table_value_as_char_v1(x)
% ============================================================
% get_table_value_as_char_v1
%
% Convert table cell/string/categorical/char value to char.
% ============================================================

    if iscell(x)
        value = x{1};
    elseif isstring(x)
        value = char(x);
    elseif iscategorical(x)
        value = char(x);
    elseif ischar(x)
        value = x;
    else
        value = char(string(x));
    end
end
function idx = make_centered_index_v1(center_idx, patch_len, max_len)
% ============================================================
% make_centered_index_v1
%
% Create a centered index range with boundary protection.
% ============================================================

    start_idx = center_idx - floor(patch_len/2) + 1;
    end_idx = start_idx + patch_len - 1;

    if start_idx < 1
        start_idx = 1;
        end_idx = patch_len;
    end

    if end_idx > max_len
        end_idx = max_len;
        start_idx = max_len - patch_len + 1;
    end

    idx = start_idx:end_idx;
end
function patch_list = make_patch_index_list_grid_v1(scan, patch_size, num_patches)
% ============================================================
% make_patch_index_list_grid_v1
%
% Generate a small fixed patch list for pilot dataset generation.
%
% patch_size = [Nz_patch, Nx_patch, Ny_patch]
% ============================================================

    Nzp = patch_size(1);
    Nxp = patch_size(2);
    Nyp = patch_size(3);

    cz = round(scan.N_z / 2);
    cx = round(scan.N_x / 2);
    cy = round(scan.N_y / 2);

    centers = [
        cz,                 cx,                 cy;
        round(0.20*scan.N_z), cx,               cy;
        round(0.78*scan.N_z), cx,               cy;
        cz,                 round(0.25*scan.N_x), round(0.25*scan.N_y);
        cz,                 round(0.75*scan.N_x), round(0.75*scan.N_y);
        cz,                 round(0.25*scan.N_x), round(0.75*scan.N_y);
        cz,                 round(0.75*scan.N_x), round(0.25*scan.N_y)
    ];

    n_available = size(centers, 1);
    n_use = min(num_patches, n_available);

    patch_list = struct('z_idx', {}, 'x_idx', {}, 'y_idx', {});

    for p = 1:n_use

        zc = centers(p, 1);
        xc = centers(p, 2);
        yc = centers(p, 3);

        patch_list(p).z_idx = make_centered_index_v1(zc, Nzp, scan.N_z);
        patch_list(p).x_idx = make_centered_index_v1(xc, Nxp, scan.N_x);
        patch_list(p).y_idx = make_centered_index_v1(yc, Nyp, scan.N_y);
    end
end
function patch_list = make_patch_index_list_dense_v1(scan, patch_size)
% ============================================================
% make_patch_index_list_dense_v1
%
% Generate a full non-overlapping tiled patch list.
%
% patch_size = [Nz_patch, Nx_patch, Ny_patch]
% stride     = patch_size
% HDF5 meta indices stay MATLAB 1-based to match existing sparse samples.
% index form matches make_centered_index_v1 output.
% ============================================================

    patch_size = double(patch_size(:)).';

    if numel(patch_size) ~= 3
        error('patch_size must be [Nz_patch, Nx_patch, Ny_patch].');
    end

    if any(patch_size <= 0) || any(mod(patch_size, 1) ~= 0)
        error('patch_size values must be positive integers.');
    end

    Nzp = patch_size(1);
    Nxp = patch_size(2);
    Nyp = patch_size(3);

    scan_size = double([scan.N_z, scan.N_x, scan.N_y]);

    if any(patch_size > scan_size)
        error('patch_size must not exceed scan size.');
    end

    if any(mod(scan_size, patch_size) ~= 0)
        error('Dense tiling requires scan size to be exactly divisible by patch_size.');
    end

    z_starts = 1:Nzp:scan.N_z;
    x_starts = 1:Nxp:scan.N_x;
    y_starts = 1:Nyp:scan.N_y;

    n_patches = numel(z_starts) * numel(x_starts) * numel(y_starts);
    patch_list = repmat(struct('z_idx', [], 'x_idx', [], 'y_idx', []), n_patches, 1);

    p = 0;

    for iz = 1:numel(z_starts)
        z0 = z_starts(iz);
        z_idx = z0:(z0 + Nzp - 1);

        for ix = 1:numel(x_starts)
            x0 = x_starts(ix);
            x_idx = x0:(x0 + Nxp - 1);

            for iy = 1:numel(y_starts)
                y0 = y_starts(iy);
                y_idx = y0:(y0 + Nyp - 1);

                p = p + 1;
                patch_list(p).z_idx = z_idx;
                patch_list(p).x_idx = x_idx;
                patch_list(p).y_idx = y_idx;
            end
        end
    end
end
function sample = make_RF_learning_sample_v2(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                             input_angle_set, target_angle_set, ...
                                             z_idx, x_idx, y_idx, source_file, frame_id)
% ============================================================
% make_RF_learning_sample_v2
%
% Generate one paired RF learning sample.
%
% input:
%   few-angle delay-aligned RF tensor
%
% baseline:
%   few-angle DAS sum and angle-averaged DAS
%
% label:
%   target-angle DAS sum and angle-averaged DAS
% ============================================================

    if nargin < 15 || isempty(source_file)
        source_file = '';
    end

    if nargin < 16 || isempty(frame_id)
        frame_id = 1;
    end

    fprintf('\n============================================================\n');
    fprintf('make_RF_learning_sample_v2\n');
    fprintf('Input angles : %d\n', length(input_angle_set));
    fprintf('Target angles: %d\n', length(target_angle_set));
    fprintf('Patch size   : Nz=%d, Nx=%d, Ny=%d\n', ...
        length(z_idx), length(x_idx), length(y_idx));
    fprintf('============================================================\n');

    % ------------------------------------------------------------
    % 1. Input RF tensor: few-angle
    % ------------------------------------------------------------
    fprintf('\n--- Extract input RF tensor ---\n');

    input_patch = extract_delay_aligned_RF_patch_v1( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan, ...
        input_angle_set, z_idx, x_idx, y_idx);

    F_RC_input = single(input_patch.F_RC_patch);
    F_CR_input = single(input_patch.F_CR_patch);

    Nz = input_patch.scan_patch.N_z;
    Nx = input_patch.scan_patch.N_x;
    Ny = input_patch.scan_patch.N_y;

    DAS_input_sum = sum(sum(F_RC_input, 5), 4) + ...
                    sum(sum(F_CR_input, 5), 4);

    DAS_input_sum = reshape(single(DAS_input_sum), [Nz, Nx, Ny]);

    DAS_input_avg = DAS_input_sum ./ single(length(input_angle_set));

    DAS_input_sum_abs = single(abs(DAS_input_sum));
    DAS_input_avg_abs = single(abs(DAS_input_avg));

    % ------------------------------------------------------------
    % 2. Target label: target-angle DAS
    % ------------------------------------------------------------
    fprintf('\n--- Extract target RF tensor for label DAS ---\n');

    target_patch = extract_delay_aligned_RF_patch_v1( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan, ...
        target_angle_set, z_idx, x_idx, y_idx);

    F_RC_target = single(target_patch.F_RC_patch);
    F_CR_target = single(target_patch.F_CR_patch);

    DAS_target_sum = sum(sum(F_RC_target, 5), 4) + ...
                     sum(sum(F_CR_target, 5), 4);

    DAS_target_sum = reshape(single(DAS_target_sum), [Nz, Nx, Ny]);

    DAS_target_avg = DAS_target_sum ./ single(length(target_angle_set));

    DAS_target_sum_abs = single(abs(DAS_target_sum));
    DAS_target_avg_abs = single(abs(DAS_target_avg));

    clear target_patch F_RC_target F_CR_target;

    % ------------------------------------------------------------
    % 3. Assemble sample
    % ------------------------------------------------------------
    sample = struct();

    sample.input = struct();
    sample.input.F_RC_patch = F_RC_input;
    sample.input.F_CR_patch = F_CR_input;

    sample.baseline = struct();
    sample.baseline.DAS_input_sum = DAS_input_sum;
    sample.baseline.DAS_input_sum_abs = DAS_input_sum_abs;
    sample.baseline.DAS_input_avg = DAS_input_avg;
    sample.baseline.DAS_input_avg_abs = DAS_input_avg_abs;

    sample.label = struct();
    sample.label.DAS_target_sum = DAS_target_sum;
    sample.label.DAS_target_sum_abs = DAS_target_sum_abs;
    sample.label.DAS_target_avg = DAS_target_avg;
    sample.label.DAS_target_avg_abs = DAS_target_avg_abs;

    sample.meta = struct();
    sample.meta.z_idx = z_idx;
    sample.meta.x_idx = x_idx;
    sample.meta.y_idx = y_idx;

    sample.meta.input_angle_set = input_angle_set;
    sample.meta.target_angle_set = target_angle_set;

    sample.meta.input_angle_count = length(input_angle_set);
    sample.meta.target_angle_count = length(target_angle_set);

    sample.meta.scan_patch = input_patch.scan_patch;

    sample.meta.x_axis_mm = input_patch.scan_patch.x_axis * 1000;
    sample.meta.y_axis_mm = input_patch.scan_patch.y_axis * 1000;
    sample.meta.z_axis_mm = input_patch.scan_patch.z_axis * 1000;

    sample.meta.source_file = source_file;
    sample.meta.frame_id = frame_id;

    sample.meta.tensor_order_input = '[Nz, Nx, Ny, N_channel, N_angle]';
    sample.meta.label_order = '[Nz, Nx, Ny]';

    sample.meta.input_sample_pos_RC_minmax = input_patch.sample_pos_RC_minmax;
    sample.meta.input_sample_pos_CR_minmax = input_patch.sample_pos_CR_minmax;
    sample.meta.input_apo_count_RC = input_patch.apo_count_RC;
    sample.meta.input_apo_count_CR = input_patch.apo_count_CR;

    % ------------------------------------------------------------
    % 4. Print summary
    % ------------------------------------------------------------
    fprintf('\n=== make_RF_learning_sample_v2 finished ===\n');
    fprintf('Input F_RC size       : [%s]\n', num2str(size(sample.input.F_RC_patch)));
    fprintf('Input F_CR size       : [%s]\n', num2str(size(sample.input.F_CR_patch)));
    fprintf('Baseline sum size     : [%s]\n', num2str(size(sample.baseline.DAS_input_sum)));
    fprintf('Baseline avg size     : [%s]\n', num2str(size(sample.baseline.DAS_input_avg)));
    fprintf('Target sum size       : [%s]\n', num2str(size(sample.label.DAS_target_sum)));
    fprintf('Target avg size       : [%s]\n', num2str(size(sample.label.DAS_target_avg)));
end

function scan = build_default_RCA_scan_v1()
% ============================================================
% build_default_RCA_scan_v1
%
% Same scan grid as the current DAS script.
% ============================================================

    pitch = 0.2e-3;

    scan = struct();

    scan.startdepth = 5e-3;
    scan.enddepth = 42e-3;

    scan.N_z = 1024;
    scan.N_x = 128;
    scan.N_y = 128;

    scan.x_axis = linspace(-127*pitch/2, 127*pitch/2, scan.N_x);
    scan.y_axis = linspace(-127*pitch/2, 127*pitch/2, scan.N_y);
    scan.z_axis = linspace(scan.startdepth, scan.enddepth, scan.N_z);

    [scan.x, scan.z, scan.y] = meshgrid(scan.x_axis, scan.z_axis, scan.y_axis);

    scan.N_pixels = numel(scan.x);
end
function [RF_Single, D, Res, Rec_S, scan] = load_RF_file_for_learning_v1(file_path)
% ============================================================
% load_RF_file_for_learning_v1
%
% Load one RF .mat file and build the default scan grid.
% ============================================================

    D = load(file_path);

    required_fields = {'Resource', 'RcvData', 'Receive', 'TX', 'TW', 'Trans'};

    for k = 1:length(required_fields)
        if ~isfield(D, required_fields{k})
            error('Missing required variable "%s" in file: %s', required_fields{k}, file_path);
        end
    end

    Res = D.Resource;

    try
        Res.RcvBuffer(1).numFrames = 1;
    catch
        % Some files may not need this.
    end

    if iscell(D.RcvData)
        RcvAll = D.RcvData{1};
    else
        RcvAll = D.RcvData;
    end

    frm = 1;

    if ndims(RcvAll) >= 3
        RF_Single = RcvAll(:, :, frm);
    else
        RF_Single = RcvAll;
    end

    Rec_S = D.Receive(1:length(D.TX));

    scan = build_default_RCA_scan_v1();
end

function log_table = generate_RF_learning_samples_from_manifest_v1( ...
    manifest_input, OutputRoot, input_angle_set, target_angle_set, patch_size, gen_opts)
% ============================================================
% generate_RF_learning_samples_from_manifest_v1
%
% Purpose:
%   Generate RF learning samples from a manifest table.
%
% One output .h5 file per learning sample.
%
% Required existing functions:
%   make_RF_learning_sample_v2
%   save_RF_learning_sample_h5_v2
%   load_RF_learning_sample_h5_v2
%   validate_RF_learning_sample_h5_v2
%   extract_delay_aligned_RF_patch_v1
% ============================================================

    if nargin < 6
        gen_opts = struct();
    end

    if ~isfield(gen_opts, 'overwrite')
        gen_opts.overwrite = false;
    end

    if ~isfield(gen_opts, 'validate_after_save')
        gen_opts.validate_after_save = true;
    end

    if ~isfield(gen_opts, 'max_files')
        gen_opts.max_files = inf;
    end

    if ~isfield(gen_opts, 'patch_grid_mode') || isempty(gen_opts.patch_grid_mode)
        gen_opts.patch_grid_mode = 'sparse';
    end

    patch_grid_mode = lower(char(string(gen_opts.patch_grid_mode)));

    if ~exist(OutputRoot, 'dir')
        mkdir(OutputRoot);
    end

    % ------------------------------------------------------------
    % 1. Load manifest
    % ------------------------------------------------------------
    if istable(manifest_input)
        manifest = manifest_input;
    else
        manifest = readtable(manifest_input);
    end

    idx_use = find(manifest.use_for_learning);

    if isfinite(gen_opts.max_files)
        idx_use = idx_use(1:min(length(idx_use), gen_opts.max_files));
    end

    fprintf('\n============================================================\n');
    fprintf('generate_RF_learning_samples_from_manifest_v1\n');
    fprintf('Files to process : %d\n', length(idx_use));
    fprintf('OutputRoot       : %s\n', OutputRoot);
    fprintf('Input angles     : %s\n', mat2str(input_angle_set));
    fprintf('Target angles    : %s\n', mat2str(target_angle_set));
    fprintf('Patch size       : [%d %d %d]\n', patch_size(1), patch_size(2), patch_size(3));
    fprintf('Patch grid mode  : %s\n', patch_grid_mode);
    fprintf('============================================================\n');

    % ------------------------------------------------------------
    % 2. Prepare log
    % ------------------------------------------------------------
    log_file_id = {};
    log_category = {};
    log_split = {};
    log_patch_id = [];
    log_file_path = {};
    log_save_path = {};
    log_status = {};
    log_message = {};
    log_elapsed_s = [];

    sample_counter = 0;

    % ------------------------------------------------------------
    % 3. Main loop over RF files
    % ------------------------------------------------------------
    for ii = 1:length(idx_use)

        row_idx = idx_use(ii);

        file_id = get_table_value_as_char_v1(manifest.file_id(row_idx));
        category = get_table_value_as_char_v1(manifest.category(row_idx));
        split_name = get_table_value_as_char_v1(manifest.split(row_idx));
        file_path = get_table_value_as_char_v1(manifest.file_path(row_idx));
        file_name = get_table_value_as_char_v1(manifest.file_name(row_idx));

        num_patches = manifest.num_patches(row_idx);

        fprintf('\n============================================================\n');
        fprintf('[File %d / %d]\n', ii, length(idx_use));
        fprintf('file_id  : %s\n', file_id);
        fprintf('category : %s\n', category);
        fprintf('split    : %s\n', split_name);
        fprintf('file     : %s\n', file_path);
        fprintf('manifest num_patches : %g\n', num_patches);
        fprintf('============================================================\n');

        try
            % ----------------------------------------------------
            % Load RF file and build scan
            % ----------------------------------------------------
            [RF_Single, D, Res, Rec_S, scan] = load_RF_file_for_learning_v1(file_path);

            % ----------------------------------------------------
            % Create patch index list
            % ----------------------------------------------------
            switch patch_grid_mode
                case {'sparse', 'sparse_v1', 'grid', 'grid_v1'}
                    patch_list = make_patch_index_list_grid_v1(scan, patch_size, num_patches);

                case {'dense', 'dense_v1', 'tile', 'tiled'}
                    patch_list = make_patch_index_list_dense_v1(scan, patch_size);

                    if isfinite(num_patches) && num_patches ~= length(patch_list)
                        fprintf('[Info] dense mode ignores manifest num_patches=%g; actual tiled patches=%d.\n', ...
                            num_patches, length(patch_list));
                    else
                        fprintf('[Info] dense mode actual tiled patches=%d.\n', ...
                            length(patch_list));
                    end

                otherwise
                    error('Unknown gen_opts.patch_grid_mode: %s', patch_grid_mode);
            end

            fprintf('actual patches  : %d\n', length(patch_list));

            % ----------------------------------------------------
            % Loop over patches
            % ----------------------------------------------------
            for p = 1:length(patch_list)

                t_patch = tic;

                z_idx = patch_list(p).z_idx;
                x_idx = patch_list(p).x_idx;
                y_idx = patch_list(p).y_idx;

                sample_counter = sample_counter + 1;

                sample_name = sprintf('%s_%s_%s_patch%03d', ...
                    file_id, category, split_name, p);

                save_dir = fullfile(OutputRoot, split_name, category);

                if ~exist(save_dir, 'dir')
                    mkdir(save_dir);
                end

                save_path = fullfile(save_dir, [sample_name, '.h5']);

                fprintf('\n--- Generate sample %s ---\n', sample_name);
                fprintf('z_idx: %d ~ %d\n', z_idx(1), z_idx(end));
                fprintf('x_idx: %d ~ %d\n', x_idx(1), x_idx(end));
                fprintf('y_idx: %d ~ %d\n', y_idx(1), y_idx(end));

                if exist(save_path, 'file') && ~gen_opts.overwrite
                    fprintf('[Skip] file exists: %s\n', save_path);

                    status = 'skipped_exists';
                    msg = 'Output file already exists.';

                else
                    source_file = file_name;
                    frame_id = 1;

                    sample = make_RF_learning_sample_v2( ...
                        RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
                        input_angle_set, target_angle_set, ...
                        z_idx, x_idx, y_idx, source_file, frame_id);

                    save_RF_learning_sample_h5_v2(sample, save_path, '/sample_000001');

                    if gen_opts.validate_after_save
                        loaded_sample = load_RF_learning_sample_h5_v2(save_path, '/sample_000001');
                        val = validate_RF_learning_sample_h5_v2(sample, loaded_sample);

                        if val.input_RC_max_abs_diff == 0 && ...
                           val.input_CR_max_abs_diff == 0 && ...
                           val.label_avg_max_abs_diff == 0
                            status = 'ok';
                            msg = 'Saved and validated.';
                        else
                            status = 'warning_nonzero_diff';
                            msg = 'Saved, but validation has nonzero difference.';
                        end
                    else
                        status = 'ok';
                        msg = 'Saved without validation.';
                    end
                end

                elapsed_s = toc(t_patch);

                log_file_id{end+1,1} = file_id;
                log_category{end+1,1} = category;
                log_split{end+1,1} = split_name;
                log_patch_id(end+1,1) = p;
                log_file_path{end+1,1} = file_path;
                log_save_path{end+1,1} = save_path;
                log_status{end+1,1} = status;
                log_message{end+1,1} = msg;
                log_elapsed_s(end+1,1) = elapsed_s;

                fprintf('[Done] status = %s, elapsed = %.1f s\n', status, elapsed_s);
            end

        catch ME
            warning('Failed to process file: %s\nError: %s', file_path, ME.message);

            log_file_id{end+1,1} = file_id;
            log_category{end+1,1} = category;
            log_split{end+1,1} = split_name;
            log_patch_id(end+1,1) = -1;
            log_file_path{end+1,1} = file_path;
            log_save_path{end+1,1} = '';
            log_status{end+1,1} = 'failed_file';
            log_message{end+1,1} = ME.message;
            log_elapsed_s(end+1,1) = NaN;
        end
    end

    % ------------------------------------------------------------
    % 4. Save generation log
    % ------------------------------------------------------------
    log_table = table( ...
        log_file_id, ...
        log_category, ...
        log_split, ...
        log_patch_id, ...
        log_file_path, ...
        log_save_path, ...
        log_status, ...
        log_message, ...
        log_elapsed_s, ...
        'VariableNames', { ...
            'file_id', ...
            'category', ...
            'split', ...
            'patch_id', ...
            'file_path', ...
            'save_path', ...
            'status', ...
            'message', ...
            'elapsed_s'});

    log_dir = fullfile(OutputRoot, '_logs');

    if ~exist(log_dir, 'dir')
        mkdir(log_dir);
    end

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    log_csv = fullfile(log_dir, ['generation_log_', timestamp, '.csv']);
    log_mat = fullfile(log_dir, ['generation_log_', timestamp, '.mat']);

    writetable(log_table, log_csv);
    save(log_mat, 'log_table');

    fprintf('\n============================================================\n');
    fprintf('Generation finished.\n');
    fprintf('Total attempted samples : %d\n', height(log_table));
    fprintf('OK samples              : %d\n', sum(strcmp(log_table.status, 'ok')));
    fprintf('Skipped samples         : %d\n', sum(strcmp(log_table.status, 'skipped_exists')));
    fprintf('Failed rows             : %d\n', sum(strcmp(log_table.status, 'failed_file')));
    fprintf('Log CSV                 : %s\n', log_csv);
    fprintf('============================================================\n');
end
function manifest = build_RF_dataset_manifest_v2_simple(DataRoot, ManifestDir)
% ============================================================
% build_RF_dataset_manifest_v2_simple
%
% Purpose:
%   Build manifest from unified RF data folders.
%
% Expected folder structure:
%   DataRoot/
%       Simu_Data/02_RF_Data/*.mat
%       Muscle_Data/02_RF_Data/*.mat
%       Carotid_Data/02_RF_Data/*.mat
%       Phantom_Data/02_RF_Data/*.mat
%
% This version does not recursively scan reconstruction folders.
% It only scans 02_RF_Data.
% ============================================================

    if nargin < 2 || isempty(ManifestDir)
        ManifestDir = fullfile(DataRoot, '_manifest');
    end

    if ~exist(ManifestDir, 'dir')
        mkdir(ManifestDir);
    end

    rng(20260521);

    category_defs = {
        'Simu_Data',    'simu_point', 100;
        'Muscle_Data',  'muscle',     100;
        'Carotid_Data', 'carotid',    100;
        'Phantom_Data', 'phantom',    100;
    };

    patches_per_file = 5;

    train_ratio = 0.70;
    val_ratio   = 0.15;

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

    for c = 1:size(category_defs, 1)

        folder_name = category_defs{c, 1};
        category_name = category_defs{c, 2};
        max_use_files = category_defs{c, 3};

        rf_folder = fullfile(DataRoot, folder_name, '02_RF_Data');

        fprintf('\nCategory: %s\n', category_name);
        fprintf('RF folder: %s\n', rf_folder);

        if ~exist(rf_folder, 'dir')
            warning('RF folder not found: %s. Skip this category.', rf_folder);
            continue;
        end

        files = dir(fullfile(rf_folder, '*.mat'));
        files = files(~[files.isdir]);

        if isempty(files)
            warning('No .mat files found in: %s', rf_folder);
            continue;
        end

        % Sort files for reproducibility, then randomize split.
        [~, order] = sort({files.name});
        files = files(order);

        n_total = length(files);
        perm = randperm(n_total);

        n_use = min(max_use_files, n_total);
        selected_idx = perm(1:n_use);
        extra_idx = perm(n_use+1:end);

        n_train = floor(train_ratio * n_use);
        n_val   = floor(val_ratio * n_use);
        n_test  = n_use - n_train - n_val;

        selected_splits = [
            repmat({'train'}, n_train, 1);
            repmat({'val'},   n_val,   1);
            repmat({'test'},  n_test,  1)
        ];

        % Learning files
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

        % Extra files
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

        fprintf('  Found files       : %d\n', n_total);
        fprintf('  Used for learning : %d\n', n_use);
        fprintf('  Train / Val / Test: %d / %d / %d\n', n_train, n_val, n_test);
        fprintf('  Extra holdout     : %d\n', length(extra_idx));
    end

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

    csv_path = fullfile(ManifestDir, 'RF_manifest_balanced_v2_simple.csv');
    mat_path = fullfile(ManifestDir, 'RF_manifest_balanced_v2_simple.mat');

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

function loaded = load_RF_learning_sample_h5_v2(save_path, sample_group)
% ============================================================
% load_RF_learning_sample_h5_v2
% ============================================================

    if nargin < 2 || isempty(sample_group)
        sample_group = '/sample_000001';
    end

    if sample_group(1) ~= '/'
        sample_group = ['/', sample_group];
    end

    if ~exist(save_path, 'file')
        error('File does not exist: %s', save_path);
    end

    loaded = struct();

    % ------------------------------------------------------------
    % 1. Input RF tensors
    % ------------------------------------------------------------
    F_RC_real = single(h5read(save_path, h5path(sample_group, 'input/F_RC_real')));
    F_RC_imag = single(h5read(save_path, h5path(sample_group, 'input/F_RC_imag')));

    F_CR_real = single(h5read(save_path, h5path(sample_group, 'input/F_CR_real')));
    F_CR_imag = single(h5read(save_path, h5path(sample_group, 'input/F_CR_imag')));

    loaded.input = struct();
    loaded.input.F_RC_patch = complex(F_RC_real, F_RC_imag);
    loaded.input.F_CR_patch = complex(F_CR_real, F_CR_imag);

    % ------------------------------------------------------------
    % 2. Baseline
    % ------------------------------------------------------------
    loaded.baseline = struct();

    r = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_sum_real')));
    im = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_sum_imag')));
    loaded.baseline.DAS_input_sum = complex(r, im);
    loaded.baseline.DAS_input_sum_abs = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_sum_abs')));

    r = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_avg_real')));
    im = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_avg_imag')));
    loaded.baseline.DAS_input_avg = complex(r, im);
    loaded.baseline.DAS_input_avg_abs = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_avg_abs')));

    % ------------------------------------------------------------
    % 3. Label
    % ------------------------------------------------------------
    loaded.label = struct();

    r = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_sum_real')));
    im = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_sum_imag')));
    loaded.label.DAS_target_sum = complex(r, im);
    loaded.label.DAS_target_sum_abs = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_sum_abs')));

    r = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_avg_real')));
    im = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_avg_imag')));
    loaded.label.DAS_target_avg = complex(r, im);
    loaded.label.DAS_target_avg_abs = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_avg_abs')));

    % ------------------------------------------------------------
    % 4. Metadata
    % ------------------------------------------------------------
    loaded.meta = struct();

    loaded.meta.z_idx = double(h5read(save_path, h5path(sample_group, 'meta/z_idx'))).';
    loaded.meta.x_idx = double(h5read(save_path, h5path(sample_group, 'meta/x_idx'))).';
    loaded.meta.y_idx = double(h5read(save_path, h5path(sample_group, 'meta/y_idx'))).';

    loaded.meta.input_angle_set = double(h5read(save_path, h5path(sample_group, 'meta/input_angle_set'))).';
    loaded.meta.target_angle_set = double(h5read(save_path, h5path(sample_group, 'meta/target_angle_set'))).';

    loaded.meta.input_angle_count = double(h5read(save_path, h5path(sample_group, 'meta/input_angle_count')));
    loaded.meta.target_angle_count = double(h5read(save_path, h5path(sample_group, 'meta/target_angle_count')));

    loaded.meta.x_axis_mm = double(h5read(save_path, h5path(sample_group, 'meta/x_axis_mm'))).';
    loaded.meta.y_axis_mm = double(h5read(save_path, h5path(sample_group, 'meta/y_axis_mm'))).';
    loaded.meta.z_axis_mm = double(h5read(save_path, h5path(sample_group, 'meta/z_axis_mm'))).';

    loaded.meta.frame_id = double(h5read(save_path, h5path(sample_group, 'meta/frame_id')));

    if has_h5_dataset(save_path, h5path(sample_group, 'meta/patch_size'))
        loaded.meta.patch_size = double(h5read(save_path, h5path(sample_group, 'meta/patch_size'))).';
    end

    if has_h5_dataset(save_path, h5path(sample_group, 'meta/input_tensor_size'))
        loaded.meta.input_tensor_size = double(h5read(save_path, h5path(sample_group, 'meta/input_tensor_size'))).';
    end

    if has_h5_dataset(save_path, h5path(sample_group, 'meta/label_size'))
        loaded.meta.label_size = double(h5read(save_path, h5path(sample_group, 'meta/label_size'))).';
    end

    try
        loaded.meta.source_file = h5readatt(save_path, h5path(sample_group, 'meta'), 'source_file');
    catch
        loaded.meta.source_file = '';
    end

    % Rebuild scan_patch
    scan_patch = struct();
    scan_patch.x_axis = loaded.meta.x_axis_mm / 1000;
    scan_patch.y_axis = loaded.meta.y_axis_mm / 1000;
    scan_patch.z_axis = loaded.meta.z_axis_mm / 1000;

    scan_patch.N_x = length(scan_patch.x_axis);
    scan_patch.N_y = length(scan_patch.y_axis);
    scan_patch.N_z = length(scan_patch.z_axis);

    [scan_patch.x, scan_patch.z, scan_patch.y] = meshgrid( ...
        scan_patch.x_axis, scan_patch.z_axis, scan_patch.y_axis);

    scan_patch.N_pixels = numel(scan_patch.x);

    loaded.meta.scan_patch = scan_patch;

    fprintf('\n=== load_RF_learning_sample_h5_v2 finished ===\n');
    fprintf('Loaded file  : %s\n', save_path);
    fprintf('Sample group : %s\n', sample_group);
    fprintf('Input F_RC   : [%s]\n', num2str(size(loaded.input.F_RC_patch)));
    fprintf('Input F_CR   : [%s]\n', num2str(size(loaded.input.F_CR_patch)));
    fprintf('Label avg    : [%s]\n', num2str(size(loaded.label.DAS_target_avg)));
    fprintf('Baseline avg : [%s]\n', num2str(size(loaded.baseline.DAS_input_avg)));
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


function audit_table = audit_RF_learning_samples_folder_v1(OutputRoot)
% ============================================================
% audit_RF_learning_samples_folder_v1
%
% Purpose:
%   Audit generated RF learning sample HDF5 files.
%
% It checks:
%   - file existence and size
%   - whether sample can be loaded
%   - input / label / baseline size
%   - NaN / Inf count
%   - category and split from folder structure
% ============================================================

    files = dir(fullfile(OutputRoot, '**', '*.h5'));
    files = files(~[files.isdir]);

    if isempty(files)
        warning('No .h5 files found under: %s', OutputRoot);
        audit_table = table();
        return;
    end

    fprintf('\n============================================================\n');
    fprintf('audit_RF_learning_samples_folder_v1\n');
    fprintf('OutputRoot : %s\n', OutputRoot);
    fprintf('H5 files   : %d\n', length(files));
    fprintf('============================================================\n');

    sample_path_list = {};
    split_list = {};
    category_list = {};
    file_size_MB = [];

    status_list = {};
    message_list = {};

    input_RC_size_list = {};
    input_CR_size_list = {};
    label_avg_size_list = {};
    baseline_avg_size_list = {};

    nan_count_list = [];
    inf_count_list = [];

    input_angle_count_list = [];
    target_angle_count_list = [];

    label_avg_mean_list = [];
    label_avg_max_list = [];
    baseline_avg_mean_list = [];
    baseline_avg_max_list = [];

    for i = 1:length(files)

        file_path = fullfile(files(i).folder, files(i).name);

        fprintf('\n[%d/%d] %s\n', i, length(files), file_path);

        try
            loaded = load_RF_learning_sample_h5_v2(file_path, '/sample_000001');

            % Parse split / category from path:
            % Expected:
            % OutputRoot/split/category/file.h5
            rel_path = erase(file_path, [OutputRoot, filesep]);
            parts = split(rel_path, filesep);

            if numel(parts) >= 3
                split_name = char(parts{1});
                category_name = char(parts{2});
            else
                split_name = '';
                category_name = '';
            end

            F_RC = loaded.input.F_RC_patch;
            F_CR = loaded.input.F_CR_patch;

            label_avg = loaded.label.DAS_target_avg;
            baseline_avg = loaded.baseline.DAS_input_avg;

            nan_count = count_nan_complex(F_RC) + ...
                        count_nan_complex(F_CR) + ...
                        count_nan_complex(label_avg) + ...
                        count_nan_complex(baseline_avg);

            inf_count = count_inf_complex(F_RC) + ...
                        count_inf_complex(F_CR) + ...
                        count_inf_complex(label_avg) + ...
                        count_inf_complex(baseline_avg);

            status = 'ok';
            msg = 'loaded';

            fprintf('  status: ok\n');
            fprintf('  category / split: %s / %s\n', category_name, split_name);
            fprintf('  F_RC size        : [%s]\n', num2str(size(F_RC)));
            fprintf('  label avg size   : [%s]\n', num2str(size(label_avg)));
            fprintf('  NaN / Inf        : %d / %d\n', nan_count, inf_count);

        catch ME
            split_name = '';
            category_name = '';

            F_RC = [];
            F_CR = [];
            label_avg = [];
            baseline_avg = [];

            nan_count = NaN;
            inf_count = NaN;

            status = 'failed';
            msg = ME.message;

            fprintf('  status: failed\n');
            fprintf('  error : %s\n', ME.message);
        end

        sample_path_list{end+1,1} = file_path;
        split_list{end+1,1} = split_name;
        category_list{end+1,1} = category_name;
        file_size_MB(end+1,1) = files(i).bytes / 1024^2;

        status_list{end+1,1} = status;
        message_list{end+1,1} = msg;

        input_RC_size_list{end+1,1} = size_to_string_v1(F_RC);
        input_CR_size_list{end+1,1} = size_to_string_v1(F_CR);
        label_avg_size_list{end+1,1} = size_to_string_v1(label_avg);
        baseline_avg_size_list{end+1,1} = size_to_string_v1(baseline_avg);

        nan_count_list(end+1,1) = nan_count;
        inf_count_list(end+1,1) = inf_count;

        if exist('loaded', 'var') && isfield(loaded, 'meta')
            input_angle_count_list(end+1,1) = loaded.meta.input_angle_count;
            target_angle_count_list(end+1,1) = loaded.meta.target_angle_count;
        else
            input_angle_count_list(end+1,1) = NaN;
            target_angle_count_list(end+1,1) = NaN;
        end

        if ~isempty(label_avg)
            label_avg_abs = abs(label_avg);
            baseline_avg_abs = abs(baseline_avg);

            label_avg_mean_list(end+1,1) = mean(label_avg_abs(:));
            label_avg_max_list(end+1,1) = max(label_avg_abs(:));
            baseline_avg_mean_list(end+1,1) = mean(baseline_avg_abs(:));
            baseline_avg_max_list(end+1,1) = max(baseline_avg_abs(:));
        else
            label_avg_mean_list(end+1,1) = NaN;
            label_avg_max_list(end+1,1) = NaN;
            baseline_avg_mean_list(end+1,1) = NaN;
            baseline_avg_max_list(end+1,1) = NaN;
        end

        clear loaded;
    end

    audit_table = table( ...
        sample_path_list, ...
        split_list, ...
        category_list, ...
        file_size_MB, ...
        status_list, ...
        message_list, ...
        input_RC_size_list, ...
        input_CR_size_list, ...
        label_avg_size_list, ...
        baseline_avg_size_list, ...
        nan_count_list, ...
        inf_count_list, ...
        input_angle_count_list, ...
        target_angle_count_list, ...
        baseline_avg_mean_list, ...
        baseline_avg_max_list, ...
        label_avg_mean_list, ...
        label_avg_max_list, ...
        'VariableNames', { ...
            'sample_path', ...
            'split', ...
            'category', ...
            'file_size_MB', ...
            'status', ...
            'message', ...
            'input_RC_size', ...
            'input_CR_size', ...
            'label_avg_size', ...
            'baseline_avg_size', ...
            'nan_count', ...
            'inf_count', ...
            'input_angle_count', ...
            'target_angle_count', ...
            'baseline_avg_mean', ...
            'baseline_avg_max', ...
            'label_avg_mean', ...
            'label_avg_max'});

    fprintf('\n============================================================\n');
    fprintf('Audit summary\n');
    fprintf('Total H5 files : %d\n', height(audit_table));
    fprintf('OK files       : %d\n', sum(strcmp(audit_table.status, 'ok')));
    fprintf('Failed files   : %d\n', sum(strcmp(audit_table.status, 'failed')));
    fprintf('Total size     : %.2f MB\n', sum(audit_table.file_size_MB));
    fprintf('============================================================\n');

    print_RF_learning_sample_audit_summary_v1(audit_table);
end

function print_RF_learning_sample_audit_summary_v1(audit_table)

    if isempty(audit_table)
        return;
    end

    fprintf('\n================ Learning Sample Audit Summary ================\n');

    categories = unique(audit_table.category, 'stable');

    for c = 1:length(categories)

        cat_name = categories{c};

        if isempty(cat_name)
            continue;
        end

        idx = strcmp(audit_table.category, cat_name);

        fprintf('\nCategory: %s\n', cat_name);
        fprintf('  samples        : %d\n', sum(idx));
        fprintf('  ok             : %d\n', sum(idx & strcmp(audit_table.status, 'ok')));
        fprintf('  failed         : %d\n', sum(idx & strcmp(audit_table.status, 'failed')));
        fprintf('  total size MB  : %.2f\n', sum(audit_table.file_size_MB(idx)));
        fprintf('  mean size MB   : %.2f\n', mean(audit_table.file_size_MB(idx)));

        fprintf('  baseline avg mean: %.3e 卤 %.3e\n', ...
            mean(audit_table.baseline_avg_mean(idx), 'omitnan'), ...
            std(audit_table.baseline_avg_mean(idx), 'omitnan'));

        fprintf('  label avg mean   : %.3e 卤 %.3e\n', ...
            mean(audit_table.label_avg_mean(idx), 'omitnan'), ...
            std(audit_table.label_avg_mean(idx), 'omitnan'));
    end

    fprintf('\nOverall:\n');
    fprintf('  samples       : %d\n', height(audit_table));
    fprintf('  ok            : %d\n', sum(strcmp(audit_table.status, 'ok')));
    fprintf('  failed        : %d\n', sum(strcmp(audit_table.status, 'failed')));
    fprintf('  total size MB : %.2f\n', sum(audit_table.file_size_MB));

    fprintf('\n===============================================================\n');
end
function s = size_to_string_v1(x)

    if isempty(x)
        s = '';
        return;
    end

    s = num2str(size(x));
end
function n = count_nan_complex(x)
    n = sum(isnan(real(x(:)))) + sum(isnan(imag(x(:))));
end

function n = count_inf_complex(x)
    n = sum(isinf(real(x(:)))) + sum(isinf(imag(x(:))));
end

function pilot_manifest = make_filelevel_pilot_manifest_v1( ...
    manifest, n_train_per_category, n_val_per_category, n_test_per_category, save_csv_path)
% ============================================================
% make_filelevel_pilot_manifest_v1
%
% Purpose:
%   Build a file-level pilot manifest.
%
% It selects:
%   n_train_per_category train RF files
%   n_val_per_category   val RF files
%   n_test_per_category  test RF files
%
% from each category.
%
% Important:
%   The split is file-level.
%   Patches from the same RF file will stay in the same split.
% ============================================================

    if nargin < 5
        save_csv_path = '';
    end

    categories = unique(manifest.category, 'stable');

    selected_rows = false(height(manifest), 1);

    fprintf('\n============================================================\n');
    fprintf('make_filelevel_pilot_manifest_v1\n');
    fprintf('Train / Val / Test per category: %d / %d / %d files\n', ...
        n_train_per_category, n_val_per_category, n_test_per_category);
    fprintf('============================================================\n');

    for c = 1:length(categories)

        cat_name = categories{c};

        idx_train = find( ...
            strcmp(manifest.category, cat_name) & ...
            strcmp(manifest.split, 'train') & ...
            manifest.use_for_learning);

        idx_val = find( ...
            strcmp(manifest.category, cat_name) & ...
            strcmp(manifest.split, 'val') & ...
            manifest.use_for_learning);

        idx_test = find( ...
            strcmp(manifest.category, cat_name) & ...
            strcmp(manifest.split, 'test') & ...
            manifest.use_for_learning);

        n_train = min(n_train_per_category, length(idx_train));
        n_val   = min(n_val_per_category,   length(idx_val));
        n_test  = min(n_test_per_category,  length(idx_test));

        selected_rows(idx_train(1:n_train)) = true;
        selected_rows(idx_val(1:n_val)) = true;
        selected_rows(idx_test(1:n_test)) = true;

        fprintf('\nCategory: %s\n', cat_name);
        fprintf('  selected train files: %d\n', n_train);
        fprintf('  selected val files  : %d\n', n_val);
        fprintf('  selected test files : %d\n', n_test);
        fprintf('  planned samples     : %d\n', ...
            5 * (n_train + n_val + n_test));
    end

    pilot_manifest = manifest(selected_rows, :);

    if ~isempty(save_csv_path)
        writetable(pilot_manifest, save_csv_path);
        fprintf('\nFile-level pilot manifest saved:\n%s\n', save_csv_path);
    end

    print_RF_manifest_summary_v1(pilot_manifest);
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

function rf_search_root = get_RF_search_root_v1(category_root, folder_name)
% ============================================================
% get_RF_search_root_v1
%
% Purpose:
%   Return the folder that should be scanned for raw RF .mat files.
%
% Priority:
%   1. category_root/02_RF_Data
%   2. category_root/Simu_Data    for Simu_Data
%   3. category_root              fallback
% ============================================================

    candidate_02 = fullfile(category_root, '02_RF_Data');

    if exist(candidate_02, 'dir')
        rf_search_root = candidate_02;
        return;
    end

    % Simu_Data may have nested Simu_Data/Simu_Data
    if strcmp(folder_name, 'Simu_Data')
        candidate_simu = fullfile(category_root, 'Simu_Data');
        if exist(candidate_simu, 'dir')
            rf_search_root = candidate_simu;
            return;
        end
    end

    rf_search_root = category_root;
end
% function files = list_rf_mat_files_recursive_v1(root_folder)
% % ============================================================
% % list_rf_mat_files_recursive_v1
% %
% % Purpose:
% %   Recursively list raw RF .mat files.
% %
% % It excludes common reconstruction result folders:
% %   03_DAS_Result
% %   03_FK_Result
% %   Recon_*
% %   NII
% %   PNG
% %   MAT under reconstruction folders
% % ============================================================
% 
%     files = dir(fullfile(root_folder, '**', '*.mat'));
% 
%     if isempty(files)
%         files = list_mat_files_recursive_fallback_v1(root_folder);
%     else
%         files = files(~[files.isdir]);
%     end
% 
%     if isempty(files)
%         return;
%     end
% 
%     keep = true(length(files), 1);
% 
%     for i = 1:length(files)
% 
%         full_path = fullfile(files(i).folder, files(i).name);
%         full_path_lower = lower(full_path);
% 
%         % Exclude reconstructed results
%         bad_patterns = {
%             lower([filesep, '03_DAS_Result', filesep])
%             lower([filesep, '03_FK_Result', filesep])
%             lower([filesep, 'DAS_Result', filesep])
%             lower([filesep, 'FK_Result', filesep])
%             lower([filesep, 'Recon_LQ', filesep])
%             lower([filesep, 'Recon_MQ', filesep])
%             lower([filesep, 'Recon_SQ', filesep])
%             lower([filesep, 'NII', filesep])
%             lower([filesep, 'PNG', filesep])
%         };
% 
%         for b = 1:length(bad_patterns)
%             if contains(full_path_lower, bad_patterns{b})
%                 keep(i) = false;
%                 break;
%             end
%         end
% 
%         % Exclude hidden files
%         if startsWith(files(i).name, '.')
%             keep(i) = false;
%         end
%     end
% 
%     files = files(keep);
% end
function files_out = filter_files_with_required_RF_vars_v1(files_in)
% ============================================================
% filter_files_with_required_RF_vars_v1
%
% Keep only files that contain the required RF variables.
% This prevents reconstructed DAS .mat files from entering manifest.
% ============================================================

    required_vars = {'Resource', 'RcvData', 'Receive', 'TX', 'TW', 'Trans'};

    keep = false(length(files_in), 1);

    for i = 1:length(files_in)

        full_path = fullfile(files_in(i).folder, files_in(i).name);

        try
            info = whos('-file', full_path);
            var_names = {info.name};

            has_all = true;

            for k = 1:length(required_vars)
                if ~ismember(required_vars{k}, var_names)
                    has_all = false;
                    break;
                end
            end

            keep(i) = has_all;

        catch
            keep(i) = false;
        end
    end

    files_out = files_in(keep);

    fprintf('  RF variable filter: kept %d / %d .mat files\n', ...
        length(files_out), length(files_in));
end
function files = list_rf_mat_files_recursive_v1(root_folder)

% ============================================================
% list_rf_mat_files_recursive_v1
%
% Purpose:
%   Recursively list raw RF .mat files.
%
% It excludes common reconstruction result folders:
%   03_DAS_Result
%   03_FK_Result
%   Recon_*
%   NII
%   PNG
%   MAT under reconstruction folders
% ============================================================

    files = dir(fullfile(root_folder, '**', '*.mat'));

    if isempty(files)
        files = list_mat_files_recursive_fallback_v1(root_folder);
    else
        files = files(~[files.isdir]);
    end

    if isempty(files)
        return;
    end

    keep = true(length(files), 1);

    for i = 1:length(files)

        full_path = fullfile(files(i).folder, files(i).name);
        full_path_lower = lower(full_path);

        % Exclude reconstructed results
        bad_patterns = {
            lower([filesep, '03_DAS_Result', filesep])
            lower([filesep, '03_FK_Result', filesep])
            lower([filesep, 'DAS_Result', filesep])
            lower([filesep, 'FK_Result', filesep])
            lower([filesep, 'Recon_LQ', filesep])
            lower([filesep, 'Recon_MQ', filesep])
            lower([filesep, 'Recon_SQ', filesep])
            lower([filesep, 'NII', filesep])
            lower([filesep, 'PNG', filesep])
        };

        for b = 1:length(bad_patterns)
            if contains(full_path_lower, bad_patterns{b})
                keep(i) = false;
                break;
            end
        end

        % Exclude hidden files
        if startsWith(files(i).name, '.')
            keep(i) = false;
        end
    end

    files = files(keep);
end
