clc; clear; close all;

% =========================================================================

% === [1. 配置区域] ===
% V1 definition:
% This function extracts delay-aligned complex RF patch tensors from RCA OPW data.
% The output tensors are the pre-summation RF samples used by DAS.
%
% F_RC_patch: [Nz, Nx, Ny, 128, N_angle], complex single
% F_CR_patch: [Nz, Nx, Ny, 128, N_angle], complex single
%
% The tensors include:
%   Hilbert-transformed RF interpolation
%   receive apodization
%   RC / CR channel mapping
%   RC / CR angle mapping
%
% The tensors do not include:
%   envelope detection
%   dB compression
%   normalization
%   real/imag splitting
%   network formatting
%   label generation
% =========================================================================
% 
% InputRoot  = 'D:\33angle\02_RF_Data';       % 输入根目录
% OutputRoot = 'D:\33angle\03_DAS_Result';    % 输出根目录

InputRoot  = 'E:\SSD\AUGAN\RF_Image';       % 输入根目录
OutputRoot = 'E:\SSD\AUGAN\RF_Image\DAS_Result';    % 输出根目录

% 任务定义
idx_SQ = 1:75;
idx_LQ = [3, 38, 73]; 

% --- [关键修改] 任务定义 ---
% 从 1 到 75 中均匀抽取 33 个角度的索引
idx_33 = round(linspace(1, 75, 33)); 

Tasks = struct('name', {}, 'indices', {}, 'folder', {});
% 现在只定义这一个任务，其他的不再重复计算
Tasks(1).name = 'MQ';                   % 名字随便起，比如 MQ (Medium Quality)
Tasks(1).indices = idx_33;              % 赋予刚刚抽取的 33 个角度
Tasks(1).folder = 'Recon_MQ_33';        % 生成的文件夹名字

if ~exist(OutputRoot, 'dir'), mkdir(OutputRoot); end

% =========================================================================
% === [2. 批量循环] ===
% =========================================================================
fileList = dir(fullfile(InputRoot, '**', '*.mat'));
fileList = fileList(~startsWith({fileList.name}, '.')); 
fprintf('--- 启动 DAS 批量处理 (带详细进度) ---\n');
fprintf('目标矩阵: 1024(Z) x 128(X) x 128(Y)\n'); 
fprintf('待处理文件数: %d\n', length(fileList));
%%
for i = 1 : length(fileList)
    fullPath = fullfile(fileList(i).folder, fileList(i).name);
    [~, baseName, ~] = fileparts(fileList(i).name);
    
    fprintf('\n======================================================\n');
    fprintf('[%d/%d] 正在处理文件: %s\n', i, length(fileList), baseName);
    fprintf('======================================================\n');
    
    % --- 加载数据 ---
    try
        D = load(fullPath); 
        Res = D.Resource; 
        Res.RcvBuffer(1).numFrames = 1; % 强制单帧
    catch ME
        fprintf('   [Error] 加载失败: %s\n', ME.message); continue; 
    end
    
    if iscell(D.RcvData), RcvAll=D.RcvData{1}; else, RcvAll=D.RcvData; end
    frm = 1; 
    RF_Single = RcvAll(:, :, frm);
    Rec_S = D.Receive(1:length(D.TX)); 
    
    % --- 准备网格 ---
    pitch = 0.2e-3;
    scan.startdepth = 5e-3;
    scan.enddepth = 42e-3; 
    
    % 强制 1024x128x128
    scan.N_z = 1024; 
    scan.N_x = 128; 
    scan.N_y = 128; 
    
    scan.x_axis = linspace(-127*pitch/2, 127*pitch/2, scan.N_x);
    scan.y_axis = linspace(-127*pitch/2, 127*pitch/2, scan.N_y);
    scan.z_axis = linspace(scan.startdepth, scan.enddepth, scan.N_z);
    
    [scan.x, scan.z, scan.y] = meshgrid(scan.x_axis, scan.z_axis, scan.y_axis);
    scan.N_pixels = numel(scan.x);
    
    % ============================================================
    % V3: make one RF learning sample
    % ============================================================
    
    input_angle_set  = [3, 38, 73];                 % 3-angle input
    target_angle_set = round(linspace(1, 75, 33));  % 33-angle label
    
    z_idx = 505:520;   % 16
    x_idx = 61:68;     % 8
    y_idx = 61:68;     % 8
    
    source_file = 'Phantom_050';
    frame_id = 1;
    
    sample = make_RF_learning_sample_v1( ...
        RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
        input_angle_set, target_angle_set, ...
        z_idx, x_idx, y_idx, source_file, frame_id);

    inspect_RF_learning_sample_v1(sample);

    save_path = fullfile('G:\DAS\Sample\', 'Phantom_050_learning_sample_000001.h5');
    
    save_RF_learning_sample_h5_v1(sample, save_path, '/sample_000001');
    
    loaded_sample = load_RF_learning_sample_h5_v1(save_path, '/sample_000001');
    
    val = validate_RF_learning_sample_h5_v1(sample, loaded_sample);
    

    return;
        

    % --- 遍历任务 (SQ / LQ) ---
    for t = 1:length(Tasks)
        currTask = Tasks(t);
        
        TaskRoot = fullfile(OutputRoot, currTask.folder);
        Dir_MAT = fullfile(TaskRoot, 'MAT');
        Dir_NII = fullfile(TaskRoot, 'NII');
        Dir_PNG = fullfile(TaskRoot, 'PNG');
        
        if ~exist(Dir_MAT, 'dir'), mkdir(Dir_MAT); end
        if ~exist(Dir_NII, 'dir'), mkdir(Dir_NII); end
        if ~exist(Dir_PNG, 'dir'), mkdir(Dir_PNG); end
        
        FileTag = sprintf('%s_%s', baseName, lower(currTask.name));
        MatPath = fullfile(Dir_MAT, [FileTag, '.mat']);
        NiiPath = fullfile(Dir_NII, [FileTag, '.nii']);
        PngPath = fullfile(Dir_PNG, [FileTag, '_Views.png']);
        
        % [增量检查]
        if exist(MatPath, 'file')
            fprintf('   -> 任务 [%s]: 跳过 (文件已存在)\n', currTask.name); continue; 
        end
        
        fprintf('   -> 任务 [%s]: 开始重建 (包含 %d 个角度)\n', currTask.name, length(currTask.indices));
        timer_recon = tic;
        
        % 1. 重建 RC (带进度条)
        vol_RC = reconstruct_RC_Phatom(RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, currTask.indices);
        
        % 2. 重建 CR (带进度条)
        vol_CR = reconstruct_CR_Phatom(RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, currTask.indices);
        
        % 3. 融合
        volume_final = vol_RC + vol_CR; 
        
        fprintf('      [Fusion] 重建完成 (耗时 %.1fs). 正在保存...\n', toc(timer_recon));
        
        % 4. 保存 MAT
        x = scan.x_axis; y = scan.y_axis; z = scan.z_axis;
        save(MatPath, 'volume_final', 'x', 'y', 'z', '-v7.3');
        
        % 5. 保存 NIfTI (dB模式: -60~0)
        vol_abs = abs(volume_final);
        max_val = max(vol_abs(:)); if max_val==0, max_val=1; end
        
        vol_db = 20 * log10(vol_abs ./ max_val + 1e-12);
        vol_db(vol_db < -60) = -60;
        vol_db(vol_db > 0)   = 0;
        
        vol_nii = single(vol_db); 
        
        dx = abs(x(2)-x(1))*1000; 
        dy = abs(y(2)-y(1))*1000;
        dz = abs(z(2)-z(1))*1000;
        vox_size = [dz, dx, dy]; 
        
        niftiwrite(vol_nii, NiiPath, 'Compressed', false);
        try
            info = niftiinfo(NiiPath);
            info.PixelDimensions = vox_size;
            info.SpaceUnits = 'Millimeter';
            if exist('affine3d', 'class')
                info.Transform = affine3d(diag([vox_size 1]));
            end
            niftiwrite(vol_nii, NiiPath, info, 'Compressed', false);
        catch
        end
        
        % 6. 保存三视图
        save_ortho_views(volume_final, scan, PngPath, FileTag);
        
    end
end
fprintf('\n全部处理完成！\n');


%%
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
% =========================================================================
% === [局部函数] 1. RC 重建 (含进度条)
% =========================================================================
function vol_RC = reconstruct_RC_Phatom(RcvData, Trans, Resource, TX, TW, Receive, scan, angle_set)
    f0 = double(Trans.frequency*1e6);
    fs = f0*Receive(1).samplesPerWave;
    c0 = 1540; 
    lambda = c0/f0;
    ElementPos = Trans.ElementPos.*lambda;
    
    num_row_waves = length(TX)/2;
    alpha = zeros(num_row_waves, 1);
    for k=1:num_row_waves, alpha(k)=TX(k).Steer(1); end
    
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, 'RC', angle_set, length(TX)/2);
    RF_data = hilbert(RF_data);
    
    CRh_probe.x = ElementPos(129:256,1);
    CRh_probe.y = ElementPos(129:256,2);
    CRh_probe.z = ElementPos(129:256,3);
    
    Cym = CRh_probe.y.' - scan.y(:);
    Czm = CRh_probe.z.' - scan.z(:);
    
    receive_delay = single(sqrt(Cym.^2+Czm.^2)/c0);
    rx_f_number = 1.5;
    Creceive_apodization = single(abs(rx_f_number.*Cym./Czm)<=0.5);
    
    D = abs(CRh_probe.y(end)-CRh_probe.y(1));
    offset_distance = TW.peak*lambda;
    initial_time = 0;
    time_vector = initial_time + (0:(size(RF_data,1)-1))/fs;
    
    vol_accum = zeros(scan.N_pixels, 1, 'single');
    
    % --- 进度显示逻辑 ---
    reverseStr = ''; % 用于退格
    total_angles = length(angle_set);
    
    for i = 1:total_angles
        % [打印进度]
        msg = sprintf('      > [RC] 计算中: 角度 %d / %d', i, total_angles);
        fprintf([reverseStr, msg]);
        reverseStr = repmat('\b', 1, length(msg));
        
        n_wave = angle_set(i);
        angle_val = alpha(n_wave);
        
        transmit_delay = scan.z(:)*cos(angle_val) + ...
                         scan.x(:)*sin(angle_val) + ...
                         (D/2)*sin(angle_val)*sign(angle_val) + ...
                         offset_distance;
                     
        for n_rx = 1:128
            delay = receive_delay(:,n_rx) + transmit_delay./c0;
            temp = Creceive_apodization(:,n_rx) .* ...
                   interp1(time_vector, RF_data(:,n_rx,i), delay, 'linear', 0);
            vol_accum = vol_accum + temp;
        end
    end
    fprintf('\n'); % 完成后换行
    vol_RC = reshape(vol_accum, [scan.N_z, scan.N_x, scan.N_y]);
end

% =========================================================================
% === [局部函数] 2. CR 重建 (含进度条)
% =========================================================================
function vol_CR = reconstruct_CR_Phatom(RcvData, Trans, Resource, TX, TW, Receive, scan, angle_set)
    f0 = double(Trans.frequency*1e6);
    fs = f0*Receive(1).samplesPerWave;
    c0 = 1540; 
    lambda = c0/f0;
    ElementPos = Trans.ElementPos.*lambda;
    
    col_waves_start = length(TX)/2;
    num_col_waves = length(TX)/2;
    beta = zeros(num_col_waves, 1);
    for k=1:num_col_waves, beta(k)=TX(col_waves_start+k).Steer(2); end
    
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, 'CR', angle_set, length(TX)/2);
    RF_data = hilbert(RF_data);
    
    RRh_probe.x = ElementPos(1:128,1);
    RRh_probe.y = ElementPos(1:128,2);
    RRh_probe.z = ElementPos(1:128,3);
    
    Rxm = RRh_probe.x.' - scan.x(:);
    Rzm = RRh_probe.z.' - scan.z(:);
    
    receive_delay = single(sqrt(Rxm.^2+Rzm.^2)/c0);
    rx_f_number = 1.5;
    Rreceive_apodization = single(abs(rx_f_number.*Rxm./Rzm)<=0.5);
    
    D = abs(RRh_probe.x(end)-RRh_probe.x(1));
    offset_distance = TW.peak*lambda;
    initial_time = 0;
    time_vector = initial_time + (0:(size(RF_data,1)-1))/fs;
    
    vol_accum = zeros(scan.N_pixels, 1, 'single');
    
    % --- 进度显示逻辑 ---
    reverseStr = '';
    total_angles = length(angle_set);
    
    for i = 1:total_angles
        % [打印进度]
        msg = sprintf('      > [CR] 计算中: 角度 %d / %d', i, total_angles);
        fprintf([reverseStr, msg]);
        reverseStr = repmat('\b', 1, length(msg));
        
        n_wave = angle_set(i);
        angle_val = beta(n_wave);
        
        transmit_delay = scan.z(:)*cos(angle_val) + ...
                         scan.y(:)*sin(angle_val) + ...
                         (D/2)*sin(angle_val)*sign(angle_val) + ...
                         offset_distance;
                     
        for n_rx = 1:128
            delay = receive_delay(:,n_rx) + transmit_delay./c0;
            temp = Rreceive_apodization(:,n_rx) .* ...
                   interp1(time_vector, RF_data(:,n_rx,i), delay, 'linear', 0);
            vol_accum = vol_accum + temp;
        end
    end
    fprintf('\n'); % 完成后换行
    vol_CR = reshape(vol_accum, [scan.N_z, scan.N_x, scan.N_y]);
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


function val = validate_delay_aligned_RF_patch_v1(out_patch, RcvData, Trans, Resource, TX, TW, Receive)
% ============================================================
% validate_delay_aligned_RF_patch_v1
%
% Purpose:
%   Validate whether the sum of delay-aligned RF tensors can reproduce
%   the original DAS patch.
%
% This function is for debugging only.
% It is intentionally separated from the RF patch extractor.
% ============================================================

    scan_patch = out_patch.scan_patch;
    angle_set = out_patch.angle_set;

    Nz = scan_patch.N_z;
    Nx = scan_patch.N_x;
    Ny = scan_patch.N_y;

    % ------------------------------------------------------------
    % 1. Sum extracted RF tensors
    % ------------------------------------------------------------
    tmp_RC = sum(sum(out_patch.F_RC_patch, 5), 4);
    tmp_CR = sum(sum(out_patch.F_CR_patch, 5), 4);

    vol_RC_from_tensor = reshape(tmp_RC, [Nz, Nx, Ny]);
    vol_CR_from_tensor = reshape(tmp_CR, [Nz, Nx, Ny]);

    vol_from_tensor = vol_RC_from_tensor + vol_CR_from_tensor;

    % ------------------------------------------------------------
    % 2. Reconstruct the same patch using original DAS functions
    % ------------------------------------------------------------
    fprintf('\n=== Original DAS patch reconstruction for validation ===\n');

    vol_RC_original = reconstruct_RC_Phatom( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan_patch, angle_set);

    vol_CR_original = reconstruct_CR_Phatom( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan_patch, angle_set);

    vol_original = vol_RC_original + vol_CR_original;

    % ------------------------------------------------------------
    % 3. Error metrics
    % ------------------------------------------------------------
    err_patch = vol_original - vol_from_tensor;

    abs_diff = abs(err_patch);
    abs_original = abs(vol_original);

    max_abs_diff = max(abs_diff(:));
    mean_abs_diff = mean(abs_diff(:));

    rel_l2_diff = norm(err_patch(:)) / (norm(vol_original(:)) + eps);
    max_rel_diff = max(abs_diff(:) ./ (abs_original(:) + eps));

    % Center voxel
    izc = ceil(Nz / 2);
    ixc = ceil(Nx / 2);
    iyc = ceil(Ny / 2);

    center_original = vol_original(izc, ixc, iyc);
    center_tensor = vol_from_tensor(izc, ixc, iyc);
    center_abs_diff = abs(center_original - center_tensor);

    % ------------------------------------------------------------
    % 4. Print validation summary
    % ------------------------------------------------------------
    fprintf('\n=== Patch validation result ===\n');
    fprintf('Patch size: Nz × Nx × Ny = %d × %d × %d\n', Nz, Nx, Ny);
    fprintf('Angles: %d\n', length(angle_set));
    fprintf('F_RC_patch size: [%s]\n', num2str(size(out_patch.F_RC_patch)));
    fprintf('F_CR_patch size: [%s]\n', num2str(size(out_patch.F_CR_patch)));

    fprintf('Max abs difference  = %.6e\n', max_abs_diff);
    fprintf('Mean abs difference = %.6e\n', mean_abs_diff);
    fprintf('Relative L2 diff    = %.6e\n', rel_l2_diff);
    fprintf('Max relative diff   = %.6e\n', max_rel_diff);

    fprintf('\nCenter voxel in patch: local [%d,%d,%d], global [%d,%d,%d]\n', ...
        izc, ixc, iyc, ...
        out_patch.z_idx(izc), out_patch.x_idx(ixc), out_patch.y_idx(iyc));

    fprintf('Original center voxel = %.6f + %.6fi\n', ...
        real(center_original), imag(center_original));

    fprintf('Tensor center voxel   = %.6f + %.6fi\n', ...
        real(center_tensor), imag(center_tensor));

    fprintf('Center abs diff       = %.6e\n', center_abs_diff);

    % ------------------------------------------------------------
    % 5. Output validation struct
    % ------------------------------------------------------------
    val = struct();

    val.scan_patch = scan_patch;
    val.angle_set = angle_set;

    val.vol_RC_from_tensor = vol_RC_from_tensor;
    val.vol_CR_from_tensor = vol_CR_from_tensor;
    val.vol_from_tensor = vol_from_tensor;

    val.vol_RC_original = vol_RC_original;
    val.vol_CR_original = vol_CR_original;
    val.vol_original = vol_original;

    val.err_patch = err_patch;

    val.max_abs_diff = max_abs_diff;
    val.mean_abs_diff = mean_abs_diff;
    val.rel_l2_diff = rel_l2_diff;
    val.max_rel_diff = max_rel_diff;

    val.center_original = center_original;
    val.center_tensor = center_tensor;
    val.center_abs_diff = center_abs_diff;

    val.out_patch = out_patch;
end


function plot_delay_aligned_patch_validation_v1(val)
% ============================================================
% plot_delay_aligned_patch_validation_v1
%
% Purpose:
%   Visualize the validation result of a delay-aligned RF patch.
%
% This function is for debugging only.
% ============================================================

    scan_patch = val.scan_patch;

    vol_original = val.vol_original;
    vol_from_tensor = val.vol_from_tensor;
    err_patch = val.err_patch;

    Nz = scan_patch.N_z;
    Ny = scan_patch.N_y;

    % Choose middle y slice
    iy_mid = ceil(Ny / 2);

    x_mm = scan_patch.x_axis * 1000;
    z_mm = scan_patch.z_axis * 1000;

    img_original = abs(squeeze(vol_original(:, :, iy_mid)));
    img_tensor   = abs(squeeze(vol_from_tensor(:, :, iy_mid)));
    img_error    = abs(squeeze(err_patch(:, :, iy_mid)));

    max_val = max([img_original(:); img_tensor(:)]);
    if max_val == 0
        max_val = 1;
    end

    img_original_db = 20 * log10(img_original ./ max_val + 1e-12);
    img_tensor_db   = 20 * log10(img_tensor   ./ max_val + 1e-12);
    img_error_db    = 20 * log10(img_error    ./ max_val + 1e-12);

    img_original_db(img_original_db < -60) = -60;
    img_tensor_db(img_tensor_db < -60) = -60;
    img_error_db(img_error_db < -60) = -60;

    % ------------------------------------------------------------
    % Figure 1: original / tensor / difference
    % ------------------------------------------------------------
    figure;

    subplot(1,3,1);
    imagesc(x_mm, z_mm, img_original_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('Original DAS patch');

    subplot(1,3,2);
    imagesc(x_mm, z_mm, img_tensor_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('Sum of RF tensor');

    subplot(1,3,3);
    imagesc(x_mm, z_mm, img_error_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('|Difference|');

    sgtitle(sprintf('Delay-aligned RF patch validation, y-slice %d / %d', iy_mid, Ny));

    % ------------------------------------------------------------
    % Figure 2: error histogram
    % ------------------------------------------------------------
    figure;
    histogram(abs(err_patch(:)), 50);
    grid on;
    xlabel('|Original - Tensor sum|');
    ylabel('Count');
    title(sprintf('Patch error histogram, Relative L2 = %.3e', val.rel_l2_diff));

    % ------------------------------------------------------------
    % Figure 3: RC / CR tensor-sum contribution
    % ------------------------------------------------------------
    figure;

    subplot(1,2,1);
    imagesc(x_mm, z_mm, 20 * log10(abs(squeeze(val.vol_RC_from_tensor(:, :, iy_mid))) ./ max_val + 1e-12));
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('RC from tensor');

    subplot(1,2,2);
    imagesc(x_mm, z_mm, 20 * log10(abs(squeeze(val.vol_CR_from_tensor(:, :, iy_mid))) ./ max_val + 1e-12));
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('CR from tensor');

    sgtitle('RC / CR tensor-sum contribution');
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

function save_RF_patch_h5_v1(out_patch, save_path)
% ============================================================
% save_RF_patch_h5_v1
%
% Purpose:
%   Save one delay-aligned RF patch to HDF5.
%
% Input:
%   out_patch.F_RC_patch : [Nz, Nx, Ny, 128, N_angle], complex single
%   out_patch.F_CR_patch : [Nz, Nx, Ny, 128, N_angle], complex single
%
% Saved datasets:
%   /F_RC_real
%   /F_RC_imag
%   /F_CR_real
%   /F_CR_imag
%
%   /DAS_real
%   /DAS_imag
%   /DAS_abs
%
%   /z_idx
%   /x_idx
%   /y_idx
%   /angle_set
%
%   /x_axis_mm
%   /y_axis_mm
%   /z_axis_mm
%
% Notes:
%   - Complex tensors are saved as real/imag single arrays.
%   - This function saves one patch per .h5 file.
%   - It does not normalize, envelope-detect, convert to dB, or split data for network.
% ============================================================

    % ------------------------------------------------------------
    % 0. Check input
    % ------------------------------------------------------------
    if ~isfield(out_patch, 'F_RC_patch') || ~isfield(out_patch, 'F_CR_patch')
        error('out_patch must contain F_RC_patch and F_CR_patch.');
    end

    if exist(save_path, 'file')
        delete(save_path);
    end

    F_RC = single(out_patch.F_RC_patch);
    F_CR = single(out_patch.F_CR_patch);

    if ~isreal(F_RC) && ~isa(F_RC, 'single')
        F_RC = single(F_RC);
    end
    if ~isreal(F_CR) && ~isa(F_CR, 'single')
        F_CR = single(F_CR);
    end

    % ------------------------------------------------------------
    % 1. Split complex tensors
    % ------------------------------------------------------------
    F_RC_real = single(real(F_RC));
    F_RC_imag = single(imag(F_RC));

    F_CR_real = single(real(F_CR));
    F_CR_imag = single(imag(F_CR));

    % ------------------------------------------------------------
    % 2. Compute DAS patch from tensor sum
    % ------------------------------------------------------------
    tmp_RC = sum(sum(F_RC, 5), 4);
    tmp_CR = sum(sum(F_CR, 5), 4);

    DAS_patch = single(tmp_RC + tmp_CR);

    DAS_real = single(real(DAS_patch));
    DAS_imag = single(imag(DAS_patch));
    DAS_abs  = single(abs(DAS_patch));

    % ------------------------------------------------------------
    % 3. Metadata
    % ------------------------------------------------------------
    z_idx = int32(out_patch.z_idx(:));
    x_idx = int32(out_patch.x_idx(:));
    y_idx = int32(out_patch.y_idx(:));
    angle_set = int32(out_patch.angle_set(:));

    x_axis_mm = single(out_patch.scan_patch.x_axis(:) * 1000);
    y_axis_mm = single(out_patch.scan_patch.y_axis(:) * 1000);
    z_axis_mm = single(out_patch.scan_patch.z_axis(:) * 1000);

    patch_size = int32([ ...
        out_patch.scan_patch.N_z; ...
        out_patch.scan_patch.N_x; ...
        out_patch.scan_patch.N_y]);

    tensor_size = int32(size(F_RC(:,:,:,:,:))).';

    % ------------------------------------------------------------
    % 4. Write main RF tensors
    % ------------------------------------------------------------
    write_h5_single_dataset(save_path, '/F_RC_real', F_RC_real);
    write_h5_single_dataset(save_path, '/F_RC_imag', F_RC_imag);
    write_h5_single_dataset(save_path, '/F_CR_real', F_CR_real);
    write_h5_single_dataset(save_path, '/F_CR_imag', F_CR_imag);

    % ------------------------------------------------------------
    % 5. Write DAS patch
    % ------------------------------------------------------------
    write_h5_single_dataset(save_path, '/DAS_real', DAS_real);
    write_h5_single_dataset(save_path, '/DAS_imag', DAS_imag);
    write_h5_single_dataset(save_path, '/DAS_abs',  DAS_abs);

    % ------------------------------------------------------------
    % 6. Write indices and axes
    % ------------------------------------------------------------
    write_h5_int32_dataset(save_path, '/z_idx', z_idx);
    write_h5_int32_dataset(save_path, '/x_idx', x_idx);
    write_h5_int32_dataset(save_path, '/y_idx', y_idx);
    write_h5_int32_dataset(save_path, '/angle_set', angle_set);

    write_h5_single_dataset(save_path, '/x_axis_mm', x_axis_mm);
    write_h5_single_dataset(save_path, '/y_axis_mm', y_axis_mm);
    write_h5_single_dataset(save_path, '/z_axis_mm', z_axis_mm);

    write_h5_int32_dataset(save_path, '/patch_size', patch_size);
    write_h5_int32_dataset(save_path, '/tensor_size', tensor_size);

    % ------------------------------------------------------------
    % 7. Optional attributes
    % ------------------------------------------------------------
    try
        h5writeatt(save_path, '/', 'format_version', 'RF_patch_h5_v1');
        h5writeatt(save_path, '/', 'description', ...
            'Delay-aligned RCA OPW RF patch. Complex tensors saved as real/imag single.');
        h5writeatt(save_path, '/', 'tensor_order', ...
            '[Nz, Nx, Ny, N_channel, N_angle]');
    catch
        warning('Could not write HDF5 attributes. Datasets were still saved.');
    end

    % ------------------------------------------------------------
    % 8. Print summary
    % ------------------------------------------------------------
    file_info = dir(save_path);

    fprintf('\n=== save_RF_patch_h5_v1 finished ===\n');
    fprintf('Saved file: %s\n', save_path);
    fprintf('F_RC size : [%s]\n', num2str(size(F_RC)));
    fprintf('F_CR size : [%s]\n', num2str(size(F_CR)));
    fprintf('DAS size  : [%s]\n', num2str(size(DAS_patch)));
    fprintf('File size : %.3f MB\n', file_info.bytes / 1024^2);
end

function loaded = load_RF_patch_h5_v1(save_path)
% ============================================================
% load_RF_patch_h5_v1
%
% Purpose:
%   Load one delay-aligned RF patch from HDF5.
%
% Output:
%   loaded.F_RC_patch : complex single
%   loaded.F_CR_patch : complex single
%   loaded.DAS_patch  : complex single
% ============================================================

    if ~exist(save_path, 'file')
        error('File does not exist: %s', save_path);
    end

    % ------------------------------------------------------------
    % 1. Load real / imag tensors
    % ------------------------------------------------------------
    F_RC_real = single(h5read(save_path, '/F_RC_real'));
    F_RC_imag = single(h5read(save_path, '/F_RC_imag'));

    F_CR_real = single(h5read(save_path, '/F_CR_real'));
    F_CR_imag = single(h5read(save_path, '/F_CR_imag'));

    loaded.F_RC_patch = complex(F_RC_real, F_RC_imag);
    loaded.F_CR_patch = complex(F_CR_real, F_CR_imag);

    % ------------------------------------------------------------
    % 2. Load DAS patch
    % ------------------------------------------------------------
    DAS_real = single(h5read(save_path, '/DAS_real'));
    DAS_imag = single(h5read(save_path, '/DAS_imag'));
    DAS_abs  = single(h5read(save_path, '/DAS_abs'));

    loaded.DAS_patch = complex(DAS_real, DAS_imag);
    loaded.DAS_abs = DAS_abs;

    % ------------------------------------------------------------
    % 3. Load metadata
    % ------------------------------------------------------------
    loaded.z_idx = double(h5read(save_path, '/z_idx')).';
    loaded.x_idx = double(h5read(save_path, '/x_idx')).';
    loaded.y_idx = double(h5read(save_path, '/y_idx')).';
    loaded.angle_set = double(h5read(save_path, '/angle_set')).';

    x_axis_mm = double(h5read(save_path, '/x_axis_mm')).';
    y_axis_mm = double(h5read(save_path, '/y_axis_mm')).';
    z_axis_mm = double(h5read(save_path, '/z_axis_mm')).';

    loaded.x_axis_mm = x_axis_mm;
    loaded.y_axis_mm = y_axis_mm;
    loaded.z_axis_mm = z_axis_mm;

    % Rebuild scan_patch
    scan_patch = struct();

    scan_patch.x_axis = x_axis_mm / 1000;
    scan_patch.y_axis = y_axis_mm / 1000;
    scan_patch.z_axis = z_axis_mm / 1000;

    scan_patch.N_x = length(scan_patch.x_axis);
    scan_patch.N_y = length(scan_patch.y_axis);
    scan_patch.N_z = length(scan_patch.z_axis);

    [scan_patch.x, scan_patch.z, scan_patch.y] = meshgrid( ...
        scan_patch.x_axis, scan_patch.z_axis, scan_patch.y_axis);

    scan_patch.N_pixels = numel(scan_patch.x);

    loaded.scan_patch = scan_patch;

    % ------------------------------------------------------------
    % 4. Optional size info
    % ------------------------------------------------------------
    if has_h5_dataset(save_path, '/patch_size')
        loaded.patch_size = double(h5read(save_path, '/patch_size')).';
    end

    if has_h5_dataset(save_path, '/tensor_size')
        loaded.tensor_size = double(h5read(save_path, '/tensor_size')).';
    end

    % ------------------------------------------------------------
    % 5. Print summary
    % ------------------------------------------------------------
    fprintf('\n=== load_RF_patch_h5_v1 finished ===\n');
    fprintf('Loaded file: %s\n', save_path);
    fprintf('F_RC size : [%s]\n', num2str(size(loaded.F_RC_patch)));
    fprintf('F_CR size : [%s]\n', num2str(size(loaded.F_CR_patch)));
    fprintf('DAS size  : [%s]\n', num2str(size(loaded.DAS_patch)));
end

function val = validate_RF_patch_h5_v1(out_patch, loaded_patch)
% ============================================================
% validate_RF_patch_h5_v1
%
% Purpose:
%   Validate whether HDF5 save/load preserves the RF patch tensors.
%
% It checks:
%   1. F_RC_patch before/after save-load
%   2. F_CR_patch before/after save-load
%   3. DAS_patch from tensor sum before/after
%   4. saved DAS_patch versus recomputed DAS_patch from loaded tensor
% ============================================================

    % ------------------------------------------------------------
    % 1. Original tensors
    % ------------------------------------------------------------
    F_RC_original = single(out_patch.F_RC_patch);
    F_CR_original = single(out_patch.F_CR_patch);

    F_RC_loaded = single(loaded_patch.F_RC_patch);
    F_CR_loaded = single(loaded_patch.F_CR_patch);

    % ------------------------------------------------------------
    % 2. Tensor equality check
    % ------------------------------------------------------------
    diff_RC = F_RC_original - F_RC_loaded;
    diff_CR = F_CR_original - F_CR_loaded;

    max_abs_diff_RC = max(abs(diff_RC(:)));
    max_abs_diff_CR = max(abs(diff_CR(:)));

    rel_l2_RC = norm(diff_RC(:)) / (norm(F_RC_original(:)) + eps);
    rel_l2_CR = norm(diff_CR(:)) / (norm(F_CR_original(:)) + eps);

    % ------------------------------------------------------------
    % 3. DAS from original tensor
    % ------------------------------------------------------------
    DAS_original = sum(sum(F_RC_original, 5), 4) + ...
                   sum(sum(F_CR_original, 5), 4);

    DAS_original = single(DAS_original);

    % ------------------------------------------------------------
    % 4. DAS from loaded tensor
    % ------------------------------------------------------------
    DAS_loaded_from_tensor = sum(sum(F_RC_loaded, 5), 4) + ...
                             sum(sum(F_CR_loaded, 5), 4);

    DAS_loaded_from_tensor = single(DAS_loaded_from_tensor);

    % ------------------------------------------------------------
    % 5. DAS saved in HDF5
    % ------------------------------------------------------------
    DAS_loaded_saved = single(loaded_patch.DAS_patch);

    diff_DAS_tensor = DAS_original - DAS_loaded_from_tensor;
    diff_DAS_saved  = DAS_loaded_from_tensor - DAS_loaded_saved;

    max_abs_diff_DAS_tensor = max(abs(diff_DAS_tensor(:)));
    rel_l2_DAS_tensor = norm(diff_DAS_tensor(:)) / (norm(DAS_original(:)) + eps);

    max_abs_diff_DAS_saved = max(abs(diff_DAS_saved(:)));
    rel_l2_DAS_saved = norm(diff_DAS_saved(:)) / (norm(DAS_loaded_from_tensor(:)) + eps);

    % ------------------------------------------------------------
    % 6. Metadata checks
    % ------------------------------------------------------------
    same_z_idx = isequal(double(out_patch.z_idx(:)).', double(loaded_patch.z_idx(:)).');
    same_x_idx = isequal(double(out_patch.x_idx(:)).', double(loaded_patch.x_idx(:)).');
    same_y_idx = isequal(double(out_patch.y_idx(:)).', double(loaded_patch.y_idx(:)).');
    same_angle_set = isequal(double(out_patch.angle_set(:)).', double(loaded_patch.angle_set(:)).');

    % ------------------------------------------------------------
    % 7. Print summary
    % ------------------------------------------------------------
    fprintf('\n=== validate_RF_patch_h5_v1 ===\n');

    fprintf('\n[RF tensor save/load check]\n');
    fprintf('RC max abs diff    = %.6e\n', max_abs_diff_RC);
    fprintf('RC relative L2     = %.6e\n', rel_l2_RC);
    fprintf('CR max abs diff    = %.6e\n', max_abs_diff_CR);
    fprintf('CR relative L2     = %.6e\n', rel_l2_CR);

    fprintf('\n[DAS recomputation check]\n');
    fprintf('Original tensor DAS vs loaded tensor DAS:\n');
    fprintf('  max abs diff     = %.6e\n', max_abs_diff_DAS_tensor);
    fprintf('  relative L2      = %.6e\n', rel_l2_DAS_tensor);

    fprintf('\n[Saved DAS check]\n');
    fprintf('Loaded tensor DAS vs saved DAS:\n');
    fprintf('  max abs diff     = %.6e\n', max_abs_diff_DAS_saved);
    fprintf('  relative L2      = %.6e\n', rel_l2_DAS_saved);

    fprintf('\n[Metadata check]\n');
    fprintf('same z_idx         = %d\n', same_z_idx);
    fprintf('same x_idx         = %d\n', same_x_idx);
    fprintf('same y_idx         = %d\n', same_y_idx);
    fprintf('same angle_set     = %d\n', same_angle_set);

    % ------------------------------------------------------------
    % 8. Output
    % ------------------------------------------------------------
    val = struct();

    val.max_abs_diff_RC = max_abs_diff_RC;
    val.max_abs_diff_CR = max_abs_diff_CR;

    val.rel_l2_RC = rel_l2_RC;
    val.rel_l2_CR = rel_l2_CR;

    val.DAS_original = DAS_original;
    val.DAS_loaded_from_tensor = DAS_loaded_from_tensor;
    val.DAS_loaded_saved = DAS_loaded_saved;

    val.max_abs_diff_DAS_tensor = max_abs_diff_DAS_tensor;
    val.rel_l2_DAS_tensor = rel_l2_DAS_tensor;

    val.max_abs_diff_DAS_saved = max_abs_diff_DAS_saved;
    val.rel_l2_DAS_saved = rel_l2_DAS_saved;

    val.same_z_idx = same_z_idx;
    val.same_x_idx = same_x_idx;
    val.same_y_idx = same_y_idx;
    val.same_angle_set = same_angle_set;
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

function tf = search_h5_dataset_recursive(group_info, dataset_name)
% Recursive helper for has_h5_dataset.

    tf = false;

    % Check datasets in current group
    for k = 1:length(group_info.Datasets)
        current_path = fullfile_h5(group_info.Name, group_info.Datasets(k).Name);

        if strcmp(current_path, dataset_name)
            tf = true;
            return;
        end
    end

    % Check subgroups
    for g = 1:length(group_info.Groups)
        tf = search_h5_dataset_recursive(group_info.Groups(g), dataset_name);
        if tf
            return;
        end
    end
end

function path_out = fullfile_h5(group_name, dataset_name)
% Build HDF5-style path.

    if strcmp(group_name, '/')
        path_out = ['/', dataset_name];
    else
        path_out = [group_name, '/', dataset_name];
    end
end
function sample = make_RF_learning_sample_v1(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                             input_angle_set, target_angle_set, ...
                                             z_idx, x_idx, y_idx, source_file, frame_id)
% ============================================================
% make_RF_learning_sample_v1
%
% Purpose:
%   Generate one paired learning sample:
%
%   input:
%       3-angle delay-aligned RF tensor
%
%   label:
%       target-angle DAS patch, e.g. 33-angle DAS
%
%   baseline:
%       input-angle DAS patch, e.g. 3-angle DAS
%
% Output:
%   sample.input.F_RC_patch        [Nz,Nx,Ny,128,N_input_angle], complex single
%   sample.input.F_CR_patch        [Nz,Nx,Ny,128,N_input_angle], complex single
%
%   sample.label.DAS_target        [Nz,Nx,Ny], complex single
%   sample.label.DAS_target_abs    [Nz,Nx,Ny], single
%
%   sample.baseline.DAS_input      [Nz,Nx,Ny], complex single
%   sample.baseline.DAS_input_abs  [Nz,Nx,Ny], single
%
% Notes:
%   - The target RF tensor is not stored in sample by default.
%   - It is only used internally to compute the target DAS patch.
% ============================================================

    if nargin < 15 || isempty(source_file)
        source_file = '';
    end

    if nargin < 16 || isempty(frame_id)
        frame_id = 1;
    end

    fprintf('\n============================================================\n');
    fprintf('make_RF_learning_sample_v1\n');
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

    % Input-angle DAS baseline
    DAS_input = sum(sum(F_RC_input, 5), 4) + ...
                sum(sum(F_CR_input, 5), 4);

    DAS_input = single(DAS_input);
    DAS_input_abs = single(abs(DAS_input));

    % ------------------------------------------------------------
    % 2. Target DAS label: target-angle
    % ------------------------------------------------------------
    fprintf('\n--- Extract target RF tensor for label DAS ---\n');

    target_patch = extract_delay_aligned_RF_patch_v1( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan, ...
        target_angle_set, z_idx, x_idx, y_idx);

    F_RC_target = single(target_patch.F_RC_patch);
    F_CR_target = single(target_patch.F_CR_patch);

    DAS_target = sum(sum(F_RC_target, 5), 4) + ...
                 sum(sum(F_CR_target, 5), 4);

    DAS_target = single(DAS_target);
    DAS_target_abs = single(abs(DAS_target));

    % Clear target RF tensor to avoid unnecessary memory retention
    clear target_patch F_RC_target F_CR_target;

    % ------------------------------------------------------------
    % 3. Assemble sample struct
    % ------------------------------------------------------------
    sample = struct();

    sample.input = struct();
    sample.input.F_RC_patch = F_RC_input;
    sample.input.F_CR_patch = F_CR_input;

    sample.label = struct();
    sample.label.DAS_target = DAS_target;
    sample.label.DAS_target_abs = DAS_target_abs;

    sample.baseline = struct();
    sample.baseline.DAS_input = DAS_input;
    sample.baseline.DAS_input_abs = DAS_input_abs;

    sample.meta = struct();

    sample.meta.z_idx = z_idx;
    sample.meta.x_idx = x_idx;
    sample.meta.y_idx = y_idx;

    sample.meta.input_angle_set = input_angle_set;
    sample.meta.target_angle_set = target_angle_set;

    sample.meta.scan_patch = input_patch.scan_patch;

    sample.meta.x_axis_mm = input_patch.scan_patch.x_axis * 1000;
    sample.meta.y_axis_mm = input_patch.scan_patch.y_axis * 1000;
    sample.meta.z_axis_mm = input_patch.scan_patch.z_axis * 1000;

    sample.meta.source_file = source_file;
    sample.meta.frame_id = frame_id;

    sample.meta.tensor_order_input = '[Nz, Nx, Ny, N_channel, N_angle]';
    sample.meta.label_order = '[Nz, Nx, Ny]';

    % Keep selected debug metadata from extractor
    sample.meta.input_sample_pos_RC_minmax = input_patch.sample_pos_RC_minmax;
    sample.meta.input_sample_pos_CR_minmax = input_patch.sample_pos_CR_minmax;
    sample.meta.input_apo_count_RC = input_patch.apo_count_RC;
    sample.meta.input_apo_count_CR = input_patch.apo_count_CR;

    % ------------------------------------------------------------
    % 4. Print summary
    % ------------------------------------------------------------
    fprintf('\n=== make_RF_learning_sample_v1 finished ===\n');
    fprintf('Input F_RC size       : [%s]\n', num2str(size(sample.input.F_RC_patch)));
    fprintf('Input F_CR size       : [%s]\n', num2str(size(sample.input.F_CR_patch)));
    fprintf('Baseline DAS size     : [%s]\n', num2str(size(sample.baseline.DAS_input)));
    fprintf('Target label DAS size : [%s]\n', num2str(size(sample.label.DAS_target)));
end

function save_RF_learning_sample_h5_v1(sample, save_path, sample_group)
% ============================================================
% save_RF_learning_sample_h5_v1
%
% Purpose:
%   Save one paired learning sample to HDF5.
%
% Structure:
%   /sample_000001/input/F_RC_real
%   /sample_000001/input/F_RC_imag
%   /sample_000001/input/F_CR_real
%   /sample_000001/input/F_CR_imag
%
%   /sample_000001/label/DAS_target_real
%   /sample_000001/label/DAS_target_imag
%   /sample_000001/label/DAS_target_abs
%
%   /sample_000001/baseline/DAS_input_real
%   /sample_000001/baseline/DAS_input_imag
%   /sample_000001/baseline/DAS_input_abs
%
%   /sample_000001/meta/...
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

    DAS_target = single(sample.label.DAS_target);
    DAS_input  = single(sample.baseline.DAS_input);

    % ------------------------------------------------------------
    % 1. Input RF tensors
    % ------------------------------------------------------------
    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_RC_real'), single(real(F_RC)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_RC_imag'), single(imag(F_RC)));

    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_CR_real'), single(real(F_CR)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'input/F_CR_imag'), single(imag(F_CR)));

    % ------------------------------------------------------------
    % 2. Target label
    % ------------------------------------------------------------
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_real'), single(real(DAS_target)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_imag'), single(imag(DAS_target)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'label/DAS_target_abs'),  single(sample.label.DAS_target_abs));

    % ------------------------------------------------------------
    % 3. Input-angle DAS baseline
    % ------------------------------------------------------------
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_real'), single(real(DAS_input)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_imag'), single(imag(DAS_input)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'baseline/DAS_input_abs'),  single(sample.baseline.DAS_input_abs));

    % ------------------------------------------------------------
    % 4. Metadata
    % ------------------------------------------------------------
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/z_idx'), int32(sample.meta.z_idx(:)));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/x_idx'), int32(sample.meta.x_idx(:)));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/y_idx'), int32(sample.meta.y_idx(:)));

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/input_angle_set'), int32(sample.meta.input_angle_set(:)));
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/target_angle_set'), int32(sample.meta.target_angle_set(:)));

    write_h5_single_dataset(save_path, h5path(sample_group, 'meta/x_axis_mm'), single(sample.meta.x_axis_mm(:)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'meta/y_axis_mm'), single(sample.meta.y_axis_mm(:)));
    write_h5_single_dataset(save_path, h5path(sample_group, 'meta/z_axis_mm'), single(sample.meta.z_axis_mm(:)));

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/frame_id'), int32(sample.meta.frame_id));

    patch_size = int32([ ...
        sample.meta.scan_patch.N_z; ...
        sample.meta.scan_patch.N_x; ...
        sample.meta.scan_patch.N_y]);

    input_tensor_size = int32(size(F_RC)).';
    label_size = int32(size(DAS_target)).';

    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/patch_size'), patch_size);
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/input_tensor_size'), input_tensor_size);
    write_h5_int32_dataset(save_path, h5path(sample_group, 'meta/label_size'), label_size);

    % Optional debug metadata
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
        h5writeatt(save_path, sample_group, 'format_version', 'RF_learning_sample_h5_v1');
        h5writeatt(save_path, sample_group, 'description', ...
            'Paired sample: few-angle delay-aligned RF tensor input and target-angle DAS label.');
        h5writeatt(save_path, sample_group, 'input_tensor_order', ...
            '[Nz, Nx, Ny, N_channel, N_angle]');
        h5writeatt(save_path, sample_group, 'label_order', ...
            '[Nz, Nx, Ny]');
        h5writeatt(save_path, h5path(sample_group, 'meta'), 'source_file', sample.meta.source_file);
    catch
        warning('Could not write one or more HDF5 attributes.');
    end

    % Print summary
    file_info = dir(save_path);

    fprintf('\n=== save_RF_learning_sample_h5_v1 finished ===\n');
    fprintf('Saved file   : %s\n', save_path);
    fprintf('Sample group : %s\n', sample_group);
    fprintf('Input F_RC   : [%s]\n', num2str(size(F_RC)));
    fprintf('Input F_CR   : [%s]\n', num2str(size(F_CR)));
    fprintf('Label DAS    : [%s]\n', num2str(size(DAS_target)));
    fprintf('Baseline DAS : [%s]\n', num2str(size(DAS_input)));
    fprintf('File size    : %.3f MB\n', file_info.bytes / 1024^2);
end


function loaded = load_RF_learning_sample_h5_v1(save_path, sample_group)

% ============================================================
% load_RF_learning_sample_h5_v1
%
% Purpose:
%   Load one paired learning sample from HDF5.
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
    % 2. Label
    % ------------------------------------------------------------
    DAS_target_real = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_real')));
    DAS_target_imag = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_imag')));
    DAS_target_abs  = single(h5read(save_path, h5path(sample_group, 'label/DAS_target_abs')));

    loaded.label = struct();
    loaded.label.DAS_target = complex(DAS_target_real, DAS_target_imag);
    loaded.label.DAS_target_abs = DAS_target_abs;

    % ------------------------------------------------------------
    % 3. Baseline
    % ------------------------------------------------------------
    DAS_input_real = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_real')));
    DAS_input_imag = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_imag')));
    DAS_input_abs  = single(h5read(save_path, h5path(sample_group, 'baseline/DAS_input_abs')));

    loaded.baseline = struct();
    loaded.baseline.DAS_input = complex(DAS_input_real, DAS_input_imag);
    loaded.baseline.DAS_input_abs = DAS_input_abs;

    % ------------------------------------------------------------
    % 4. Metadata
    % ------------------------------------------------------------
    loaded.meta = struct();

    loaded.meta.z_idx = double(h5read(save_path, h5path(sample_group, 'meta/z_idx'))).';
    loaded.meta.x_idx = double(h5read(save_path, h5path(sample_group, 'meta/x_idx'))).';
    loaded.meta.y_idx = double(h5read(save_path, h5path(sample_group, 'meta/y_idx'))).';

    loaded.meta.input_angle_set = double(h5read(save_path, h5path(sample_group, 'meta/input_angle_set'))).';
    loaded.meta.target_angle_set = double(h5read(save_path, h5path(sample_group, 'meta/target_angle_set'))).';

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

    % Print summary
    fprintf('\n=== load_RF_learning_sample_h5_v1 finished ===\n');
    fprintf('Loaded file  : %s\n', save_path);
    fprintf('Sample group : %s\n', sample_group);
    fprintf('Input F_RC   : [%s]\n', num2str(size(loaded.input.F_RC_patch)));
    fprintf('Input F_CR   : [%s]\n', num2str(size(loaded.input.F_CR_patch)));
    fprintf('Label DAS    : [%s]\n', num2str(size(loaded.label.DAS_target)));
    fprintf('Baseline DAS : [%s]\n', num2str(size(loaded.baseline.DAS_input)));
end
function val = validate_RF_learning_sample_h5_v1(sample, loaded)
% ============================================================
% validate_RF_learning_sample_h5_v1
%
% Purpose:
%   Validate that HDF5 save/load preserves the learning sample.
% ============================================================

    val = struct();

    % ------------------------------------------------------------
    % 1. Input tensor checks
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
    % 2. Label checks
    % ------------------------------------------------------------
    diff_label = single(sample.label.DAS_target) - single(loaded.label.DAS_target);
    diff_label_abs = single(sample.label.DAS_target_abs) - single(loaded.label.DAS_target_abs);

    val.label_max_abs_diff = max(abs(diff_label(:)));
    val.label_rel_l2 = norm(diff_label(:)) / ...
        (norm(single(sample.label.DAS_target(:))) + eps);

    val.label_abs_max_abs_diff = max(abs(diff_label_abs(:)));

    % ------------------------------------------------------------
    % 3. Baseline checks
    % ------------------------------------------------------------
    diff_baseline = single(sample.baseline.DAS_input) - single(loaded.baseline.DAS_input);
    diff_baseline_abs = single(sample.baseline.DAS_input_abs) - single(loaded.baseline.DAS_input_abs);

    val.baseline_max_abs_diff = max(abs(diff_baseline(:)));
    val.baseline_rel_l2 = norm(diff_baseline(:)) / ...
        (norm(single(sample.baseline.DAS_input(:))) + eps);

    val.baseline_abs_max_abs_diff = max(abs(diff_baseline_abs(:)));

    % ------------------------------------------------------------
    % 4. Recompute baseline from loaded input tensor
    % ------------------------------------------------------------
    DAS_input_from_loaded_RF = sum(sum(loaded.input.F_RC_patch, 5), 4) + ...
                               sum(sum(loaded.input.F_CR_patch, 5), 4);

    DAS_input_from_loaded_RF = single(DAS_input_from_loaded_RF);

    diff_recomputed_baseline = DAS_input_from_loaded_RF - single(loaded.baseline.DAS_input);

    val.loaded_RF_vs_loaded_baseline_max_abs_diff = max(abs(diff_recomputed_baseline(:)));
    val.loaded_RF_vs_loaded_baseline_rel_l2 = norm(diff_recomputed_baseline(:)) / ...
        (norm(single(loaded.baseline.DAS_input(:))) + eps);

    % ------------------------------------------------------------
    % 5. Metadata checks
    % ------------------------------------------------------------
    val.same_z_idx = isequal(double(sample.meta.z_idx(:)).', double(loaded.meta.z_idx(:)).');
    val.same_x_idx = isequal(double(sample.meta.x_idx(:)).', double(loaded.meta.x_idx(:)).');
    val.same_y_idx = isequal(double(sample.meta.y_idx(:)).', double(loaded.meta.y_idx(:)).');

    val.same_input_angle_set = isequal(double(sample.meta.input_angle_set(:)).', ...
                                       double(loaded.meta.input_angle_set(:)).');

    val.same_target_angle_set = isequal(double(sample.meta.target_angle_set(:)).', ...
                                        double(loaded.meta.target_angle_set(:)).');

    % ------------------------------------------------------------
    % 6. Print summary
    % ------------------------------------------------------------
    fprintf('\n=== validate_RF_learning_sample_h5_v1 ===\n');

    fprintf('\n[Input RF tensor]\n');
    fprintf('F_RC max abs diff = %.6e\n', val.input_RC_max_abs_diff);
    fprintf('F_RC rel L2       = %.6e\n', val.input_RC_rel_l2);
    fprintf('F_CR max abs diff = %.6e\n', val.input_CR_max_abs_diff);
    fprintf('F_CR rel L2       = %.6e\n', val.input_CR_rel_l2);

    fprintf('\n[Label DAS target]\n');
    fprintf('DAS target max abs diff     = %.6e\n', val.label_max_abs_diff);
    fprintf('DAS target rel L2           = %.6e\n', val.label_rel_l2);
    fprintf('DAS target abs max abs diff = %.6e\n', val.label_abs_max_abs_diff);

    fprintf('\n[Input-angle DAS baseline]\n');
    fprintf('DAS input max abs diff      = %.6e\n', val.baseline_max_abs_diff);
    fprintf('DAS input rel L2            = %.6e\n', val.baseline_rel_l2);
    fprintf('DAS input abs max abs diff  = %.6e\n', val.baseline_abs_max_abs_diff);

    fprintf('\n[Loaded input RF -> loaded baseline check]\n');
    fprintf('max abs diff = %.6e\n', val.loaded_RF_vs_loaded_baseline_max_abs_diff);
    fprintf('rel L2       = %.6e\n', val.loaded_RF_vs_loaded_baseline_rel_l2);

    fprintf('\n[Metadata]\n');
    fprintf('same z_idx             = %d\n', val.same_z_idx);
    fprintf('same x_idx             = %d\n', val.same_x_idx);
    fprintf('same y_idx             = %d\n', val.same_y_idx);
    fprintf('same input_angle_set   = %d\n', val.same_input_angle_set);
    fprintf('same target_angle_set  = %d\n', val.same_target_angle_set);
end

function inspect_RF_learning_sample_v1(sample)
% ============================================================
% inspect_RF_learning_sample_v1
%
% Purpose:
%   Inspect one RF learning sample:
%   input RF tensor, 3-angle DAS baseline, and target-angle DAS label.
%
% This function does not modify or save anything.
% ============================================================

    F_RC = sample.input.F_RC_patch;
    F_CR = sample.input.F_CR_patch;

    DAS_input  = sample.baseline.DAS_input;
    DAS_target = sample.label.DAS_target;

    DAS_input_abs  = abs(DAS_input);
    DAS_target_abs = abs(DAS_target);

    diff_abs = abs(DAS_target - DAS_input);

    fprintf('\n=== inspect_RF_learning_sample_v1 ===\n');

    fprintf('Input F_RC size       : [%s]\n', num2str(size(F_RC)));
    fprintf('Input F_CR size       : [%s]\n', num2str(size(F_CR)));
    fprintf('Baseline DAS size     : [%s]\n', num2str(size(DAS_input)));
    fprintf('Target label DAS size : [%s]\n', num2str(size(DAS_target)));

    fprintf('\n[NaN / Inf check]\n');
    fprintf('F_RC NaN count        : %d\n', sum(isnan(real(F_RC(:)))) + sum(isnan(imag(F_RC(:)))));
    fprintf('F_CR NaN count        : %d\n', sum(isnan(real(F_CR(:)))) + sum(isnan(imag(F_CR(:)))));
    fprintf('DAS_input NaN count   : %d\n', sum(isnan(real(DAS_input(:)))) + sum(isnan(imag(DAS_input(:)))));
    fprintf('DAS_target NaN count  : %d\n', sum(isnan(real(DAS_target(:)))) + sum(isnan(imag(DAS_target(:)))));

    fprintf('F_RC Inf count        : %d\n', sum(isinf(real(F_RC(:)))) + sum(isinf(imag(F_RC(:)))));
    fprintf('F_CR Inf count        : %d\n', sum(isinf(real(F_CR(:)))) + sum(isinf(imag(F_CR(:)))));
    fprintf('DAS_input Inf count   : %d\n', sum(isinf(real(DAS_input(:)))) + sum(isinf(imag(DAS_input(:)))));
    fprintf('DAS_target Inf count  : %d\n', sum(isinf(real(DAS_target(:)))) + sum(isinf(imag(DAS_target(:)))));

    fprintf('\n[Amplitude statistics]\n');
    fprintf('|F_RC|     min / mean / max = %.3e / %.3e / %.3e\n', ...
        min(abs(F_RC(:))), mean(abs(F_RC(:))), max(abs(F_RC(:))));
    fprintf('|F_CR|     min / mean / max = %.3e / %.3e / %.3e\n', ...
        min(abs(F_CR(:))), mean(abs(F_CR(:))), max(abs(F_CR(:))));
    fprintf('|DAS3|     min / mean / max = %.3e / %.3e / %.3e\n', ...
        min(DAS_input_abs(:)), mean(DAS_input_abs(:)), max(DAS_input_abs(:)));
    fprintf('|DASlabel| min / mean / max = %.3e / %.3e / %.3e\n', ...
        min(DAS_target_abs(:)), mean(DAS_target_abs(:)), max(DAS_target_abs(:)));
    fprintf('|label - baseline| mean / max = %.3e / %.3e\n', ...
        mean(diff_abs(:)), max(diff_abs(:)));

    % ============================================================
    % Visualization
    % ============================================================

    scan_patch = sample.meta.scan_patch;

    x_mm = scan_patch.x_axis * 1000;
    z_mm = scan_patch.z_axis * 1000;

    Ny = scan_patch.N_y;
    iy_mid = ceil(Ny / 2);

    img_input  = abs(squeeze(DAS_input(:, :, iy_mid)));
    img_target = abs(squeeze(DAS_target(:, :, iy_mid)));
    img_diff   = abs(squeeze(DAS_target(:, :, iy_mid) - DAS_input(:, :, iy_mid)));

    max_val = max([img_input(:); img_target(:)]);
    if max_val == 0
        max_val = 1;
    end

    img_input_db  = 20 * log10(img_input  ./ max_val + 1e-12);
    img_target_db = 20 * log10(img_target ./ max_val + 1e-12);
    img_diff_db   = 20 * log10(img_diff   ./ max_val + 1e-12);

    img_input_db(img_input_db < -60) = -60;
    img_target_db(img_target_db < -60) = -60;
    img_diff_db(img_diff_db < -60) = -60;

    figure;

    subplot(1,3,1);
    imagesc(x_mm, z_mm, img_input_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('3-angle DAS baseline');

    subplot(1,3,2);
    imagesc(x_mm, z_mm, img_target_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('33-angle DAS label');

    subplot(1,3,3);
    imagesc(x_mm, z_mm, img_diff_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('|Label - Baseline|');

    sgtitle(sprintf('Learning sample inspection, y-slice %d / %d', iy_mid, Ny));

    % ============================================================
    % Center voxel channel-angle input map
    % ============================================================

    Nz = scan_patch.N_z;
    Nx = scan_patch.N_x;
    Ny = scan_patch.N_y;

    izc = ceil(Nz / 2);
    ixc = ceil(Nx / 2);
    iyc = ceil(Ny / 2);

    F_RC_center = squeeze(abs(F_RC(izc, ixc, iyc, :, :)));  % [128, N_angle]
    F_CR_center = squeeze(abs(F_CR(izc, ixc, iyc, :, :)));  % [128, N_angle]

    figure;

    subplot(1,2,1);
    imagesc(F_RC_center);
    axis xy;
    colorbar;
    xlabel('Input angle index');
    ylabel('Receive channel');
    title('Center voxel |F\_RC(channel, angle)|');

    subplot(1,2,2);
    imagesc(F_CR_center);
    axis xy;
    colorbar;
    xlabel('Input angle index');
    ylabel('Receive channel');
    title('Center voxel |F\_CR(channel, angle)|');

    sgtitle(sprintf('Center voxel input RF tensor, local [%d,%d,%d]', izc, ixc, iyc));
end
