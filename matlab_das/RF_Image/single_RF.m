clc; clear; close all;

% =========================================================================

% === [1. 配置区域] ===

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
    % Debug: 单个体素、单个 RC 角度、单个接收通道
    % ============================================================
    
    angle_set_debug = [3, 38, 73];  % 先用 3 个角度
    i_angle = 2;                    % 使用 angle_set_debug 里的第 2 个，也就是 38
    n_rx = 64;                      % 第 64 个接收通道
    
    iz = 512;
    ix = 64;
    iy = 64;
    
    out = debug_one_RC_sample( ...
        RF_Single, ...
        D.Trans, ...
        Res, ...
        D.TX, ...
        D.TW, ...
        Rec_S, ...
        scan, ...
        angle_set_debug, ...
        i_angle, ...
        n_rx, ...
        iz, ...
        ix, ...
        iy);
    
    disp(out);
    
    debug_plot_one_RC_sample( ...
    RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
    angle_set_debug, i_angle, n_rx, iz, ix, iy);

    
    debug_plot_RC_all_channels_one_voxel( ...
    RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
    angle_set_debug, i_angle, iz, ix, iy);

    out_angles = debug_plot_RC_all_angles_one_voxel( ...
    RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
    angle_set_debug, iz, ix, iy);
    
    out_CR_angles = debug_plot_CR_all_angles_one_voxel( ...
    RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
    angle_set_debug, iz, ix, iy);
    % ============================================================
    % Validation: 用原 reconstruct_CR_Phatom 只重建一个 voxel
    % ============================================================
    
    scan_one = struct();
    
    scan_one.N_z = 1;
    scan_one.N_x = 1;
    scan_one.N_y = 1;
    
    scan_one.x_axis = scan.x_axis(ix);
    scan_one.y_axis = scan.y_axis(iy);
    scan_one.z_axis = scan.z_axis(iz);
    
    [scan_one.x, scan_one.z, scan_one.y] = meshgrid( ...
        scan_one.x_axis, scan_one.z_axis, scan_one.y_axis);
    
    scan_one.N_pixels = numel(scan_one.x);
    
    vol_CR_one = reconstruct_CR_Phatom( ...
        RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan_one, angle_set_debug);
    
    CR_from_original_function = vol_CR_one(1,1,1);
    CR_from_aligned_tensor = out_CR_angles.CR_total_sum;
    
    fprintf('\n=== CR single-voxel validation ===\n');
    fprintf('Original reconstruct_CR_Phatom = %.6f + %.6fi\n', ...
            real(CR_from_original_function), imag(CR_from_original_function));
    
    fprintf('Sum of aligned RF tensor     = %.6f + %.6fi\n', ...
            real(CR_from_aligned_tensor), imag(CR_from_aligned_tensor));
    
    fprintf('Abs difference               = %.6e\n', ...
            abs(CR_from_original_function - CR_from_aligned_tensor));
    
    fprintf('Relative difference          = %.6e\n', ...
            abs(CR_from_original_function - CR_from_aligned_tensor) / ...
            (abs(CR_from_original_function) + eps));
    % ============================================================
    % Validation: 用原 reconstruct_RC_Phatom 只重建一个 voxel
    % ============================================================
    
    scan_one = struct();
    
    scan_one.N_z = 1;
    scan_one.N_x = 1;
    scan_one.N_y = 1;
    
    scan_one.x_axis = scan.x_axis(ix);
    scan_one.y_axis = scan.y_axis(iy);
    scan_one.z_axis = scan.z_axis(iz);
    
    [scan_one.x, scan_one.z, scan_one.y] = meshgrid( ...
        scan_one.x_axis, scan_one.z_axis, scan_one.y_axis);
    
    scan_one.N_pixels = numel(scan_one.x);
    
    vol_RC_one = reconstruct_RC_Phatom( ...
        RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan_one, angle_set_debug);
    
    RC_from_original_function = vol_RC_one(1,1,1);
    RC_from_aligned_tensor = out_angles.RC_total_sum;
    
    fprintf('\n=== RC single-voxel validation ===\n');
    fprintf('Original reconstruct_RC_Phatom = %.6f + %.6fi\n', ...
            real(RC_from_original_function), imag(RC_from_original_function));
    
    fprintf('Sum of aligned RF tensor     = %.6f + %.6fi\n', ...
            real(RC_from_aligned_tensor), imag(RC_from_aligned_tensor));
    
    fprintf('Abs difference               = %.6e\n', ...
            abs(RC_from_original_function - RC_from_aligned_tensor));
    
    fprintf('Relative difference          = %.6e\n', ...
            abs(RC_from_original_function - RC_from_aligned_tensor) / ...
            (abs(RC_from_original_function) + eps));

    % ============================================================
    % Validation: RC + CR single-voxel fusion
    % ============================================================
    
    DAS_from_original_function = RC_from_original_function + CR_from_original_function;
    DAS_from_aligned_tensor    = RC_from_aligned_tensor    + CR_from_aligned_tensor;
    
    fprintf('\n=== RC + CR single-voxel fusion validation ===\n');
    
    fprintf('Original RC + CR DAS voxel = %.6f + %.6fi\n', ...
            real(DAS_from_original_function), imag(DAS_from_original_function));
    
    fprintf('Aligned tensor RC + CR     = %.6f + %.6fi\n', ...
            real(DAS_from_aligned_tensor), imag(DAS_from_aligned_tensor));
    
    fprintf('Abs difference             = %.6e\n', ...
            abs(DAS_from_original_function - DAS_from_aligned_tensor));
    
    fprintf('Relative difference        = %.6e\n', ...
            abs(DAS_from_original_function - DAS_from_aligned_tensor) / ...
            (abs(DAS_from_original_function) + eps));
    
    fprintf('Original abs               = %.6f\n', abs(DAS_from_original_function));
    fprintf('Aligned tensor abs         = %.6f\n', abs(DAS_from_aligned_tensor));



    % ============================================================
    % Debug: 小 patch 的 delay-aligned RF tensor 验证
    % ============================================================
    
    angle_set_debug = [3, 38, 73];
    
    % 以刚才的单体素为中心附近，做一个 8 × 4 × 4 patch
    z_idx = 509:516;   % Nz_patch = 8
    x_idx = 63:66;     % Nx_patch = 4
    y_idx = 63:66;     % Ny_patch = 4
    
    out_patch = extract_aligned_patch_RC_CR_debug( ...
        RF_Single, D.Trans, Res, D.TX, D.TW, Rec_S, scan, ...
        angle_set_debug, z_idx, x_idx, y_idx);
    
    plot_patch_validation_debug(out_patch);
    
    return;  % 先只跑 patch debug，不进入完整批处理
%     return;  % 先只跑 debug，不进入后面的完整 DAS 批处理


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

% =========================================================================
% === [局部函数] 3. 数据提取 (含 Trans.Connector)
% =========================================================================
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

% =========================================================================
% === [局部函数] 4. 保存三视图
% =========================================================================
function save_ortho_views(vol_complex, scan, save_path, title_str)
    vol_abs = abs(vol_complex);
    max_val = max(vol_abs(:)); if max_val==0, max_val=1; end
    vol_db = 20*log10(vol_abs ./ max_val);
    db_min = -60; db_max = 0;
    
    h = figure('Visible', 'off'); 
    set(h, 'Position', [100 100 1200 400]);
    sgtitle(strrep(title_str, '_', '\_'), 'FontSize', 12);
    
    z_idx = round(scan.N_z / 2); 
    img_xy = squeeze(vol_db(z_idx, :, :));
    subplot(1,3,1);
    imagesc(scan.x_axis*1000, scan.y_axis*1000, img_xy);
    title(sprintf('XY (Z=%.1fmm)', scan.z_axis(z_idx)*1000));
    xlabel('Lateral [mm]'); ylabel('Elevation [mm]');
    axis image; colormap gray; caxis([db_min db_max]); colorbar;
    
    y_idx = round(scan.N_y / 2); 
    img_xz = squeeze(vol_db(:, :, y_idx));
    subplot(1,3,2);
    imagesc(scan.x_axis*1000, scan.z_axis*1000, img_xz);
    title(sprintf('XZ (Y=%.1fmm)', scan.y_axis(y_idx)*1000));
    xlabel('Lateral [mm]'); ylabel('Axial [mm]');
    axis image; colormap gray; caxis([db_min db_max]); colorbar;
    
    x_idx = round(scan.N_x / 2); 
    img_yz = squeeze(vol_db(:, x_idx, :));
    subplot(1,3,3);
    imagesc(scan.y_axis*1000, scan.z_axis*1000, img_yz);
    title(sprintf('YZ (X=%.1fmm)', scan.x_axis(x_idx)*1000));
    xlabel('Elevation [mm]'); ylabel('Axial [mm]');
    axis image; colormap gray; caxis([db_min db_max]); colorbar;
    
    saveas(h, save_path);
    close(h);
end

function out = debug_one_RC_sample(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                   angle_set, i_angle, n_rx, iz, ix, iy)

    % === 1. 基本参数 ===
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;
    c0 = 1540;
    lambda = c0 / f0;
    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;

    % === 2. 提取 RC RF 数据 ===
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                                'RC', angle_set, half_waves);

    % 与原 DAS 保持一致：先 Hilbert，再插值
    RF_data = hilbert(RF_data);

    % === 3. 当前体素坐标 ===
    x0 = scan.x_axis(ix);
    y0 = scan.y_axis(iy);
    z0 = scan.z_axis(iz);

    % === 4. 当前 RC 角度 ===
    n_wave = angle_set(i_angle);
    angle_val = TX(n_wave).Steer(1);

    % === 5. 当前 RC 接收通道位置 ===
    CRh_probe.x = ElementPos(129:256, 1);
    CRh_probe.y = ElementPos(129:256, 2);
    CRh_probe.z = ElementPos(129:256, 3);

    rx_x = CRh_probe.x(n_rx);
    rx_y = CRh_probe.y(n_rx);
    rx_z = CRh_probe.z(n_rx);

    % === 6. RC 接收时间 ===
    receive_distance = sqrt((rx_y - y0)^2 + (rx_z - z0)^2);
    receive_time = receive_distance / c0;

    % === 7. RC 发射时间 ===
    D = abs(CRh_probe.y(end) - CRh_probe.y(1));
    offset_distance = TW.peak * lambda;

    transmit_distance = z0 * cos(angle_val) + ...
                        x0 * sin(angle_val) + ...
                        (D/2) * sin(angle_val) * sign(angle_val) + ...
                        offset_distance;

    transmit_time = transmit_distance / c0;

    % === 8. 总 delay ===
    delay = receive_time + transmit_time;

    % RF 时间轴
    initial_time = 0;
    time_vector = initial_time + (0:(size(RF_data,1)-1)) / fs;

    % 这个值表示 delay 落在 RF 的第几个采样点附近
    sample_position = delay * fs + 1;

    % === 9. 插值取 RF sample ===
    sample = interp1(time_vector, RF_data(:, n_rx, i_angle), ...
                     delay, 'linear', 0);

    % === 10. 接收孔径 apodization ===
    rx_f_number = 1.5;
    apo = single(abs(rx_f_number * (rx_y - y0) / (rx_z - z0)) <= 0.5);

    aligned_sample = apo * sample;

    % === 11. 输出结果 ===
    out = struct();

    out.iz_ix_iy = [iz, ix, iy];
    out.x0_y0_z0_m = [x0, y0, z0];
    out.x0_y0_z0_mm = [x0, y0, z0] * 1000;

    out.angle_index_in_angle_set = i_angle;
    out.n_wave = n_wave;
    out.angle_rad = angle_val;
    out.angle_deg = angle_val * 180 / pi;

    out.n_rx = n_rx;
    out.element_id = 128 + n_rx;
    out.rx_position_m = [rx_x, rx_y, rx_z];
    out.rx_position_mm = [rx_x, rx_y, rx_z] * 1000;

    out.receive_distance_m = receive_distance;
    out.receive_time_us = receive_time * 1e6;

    out.transmit_distance_m = transmit_distance;
    out.transmit_time_us = transmit_time * 1e6;

    out.delay_us = delay * 1e6;
    out.sample_position = sample_position;

    out.apodization = apo;

    out.sample_real = real(sample);
    out.sample_imag = imag(sample);
    out.sample_abs = abs(sample);

    out.aligned_sample_real = real(aligned_sample);
    out.aligned_sample_imag = imag(aligned_sample);
    out.aligned_sample_abs = abs(aligned_sample);
end

function debug_plot_one_RC_sample(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                  angle_set, i_angle, n_rx, iz, ix, iy)

    % === 1. 基本参数 ===
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;
    c0 = 1540;
    lambda = c0 / f0;
    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;

    % === 2. 提取 RC RF 数据，与 DAS 保持一致 ===
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                                'RC', angle_set, half_waves);
    RF_data = hilbert(RF_data);

    % === 3. 体素坐标 ===
    x0 = scan.x_axis(ix);
    y0 = scan.y_axis(iy);
    z0 = scan.z_axis(iz);

    % === 4. RC 角度 ===
    n_wave = angle_set(i_angle);
    angle_val = TX(n_wave).Steer(1);

    % === 5. RC 接收阵元位置 ===
    CRh_probe.x = ElementPos(129:256, 1);
    CRh_probe.y = ElementPos(129:256, 2);
    CRh_probe.z = ElementPos(129:256, 3);

    rx_y = CRh_probe.y(n_rx);
    rx_z = CRh_probe.z(n_rx);

    % === 6. delay 计算 ===
    receive_distance = sqrt((rx_y - y0)^2 + (rx_z - z0)^2);
    receive_time = receive_distance / c0;

    D = abs(CRh_probe.y(end) - CRh_probe.y(1));
    offset_distance = TW.peak * lambda;

    transmit_distance = z0 * cos(angle_val) + ...
                        x0 * sin(angle_val) + ...
                        (D/2) * sin(angle_val) * sign(angle_val) + ...
                        offset_distance;

    transmit_time = transmit_distance / c0;
    delay = receive_time + transmit_time;

    time_vector = (0:(size(RF_data,1)-1)) / fs;
    sample_position = delay * fs + 1;

    % === 7. 取当前 RF 通道 ===
    rf_trace = RF_data(:, n_rx, i_angle);
    rf_abs = abs(rf_trace);

    sample = interp1(time_vector, rf_trace, delay, 'linear', 0);

    % === 8. 只画 delay 附近一小段，方便看清楚 ===
    center_idx = round(sample_position);
    win = 80;

    idx1 = max(center_idx - win, 1);
    idx2 = min(center_idx + win, length(rf_trace));

    t_us = time_vector(idx1:idx2) * 1e6;
    rf_local = rf_abs(idx1:idx2);

    figure;
    plot(t_us, rf_local, 'LineWidth', 1.2); hold on;
    xline(delay * 1e6, '--', 'LineWidth', 1.5);

    plot(delay * 1e6, abs(sample), 'o', 'MarkerSize', 8, 'LineWidth', 1.5);

    grid on;
    xlabel('Time [\mus]');
    ylabel('|Analytic RF|');
    title(sprintf('RC single-channel RF sampling: angle %.1f deg, rx %d, sample %.2f', ...
          angle_val * 180/pi, n_rx, sample_position));

    legend('|RF(t)|', 'delay position', 'interpolated sample');
end


function out = debug_plot_RC_all_channels_one_voxel(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                                    angle_set, i_angle, iz, ix, iy)

    % ============================================================
    % 目的：
    % 同一个 voxel、同一个 RC angle 下，
    % 看 128 个接收通道的 RF 图像，以及每个通道对应的 delay 位置。
    % ============================================================

    % === 1. 基本参数 ===
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;
    c0 = 1540;
    lambda = c0 / f0;
    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;

    % === 2. 提取 RC RF 数据，与 DAS 保持一致 ===
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                                'RC', angle_set, half_waves);

    RF_data = hilbert(RF_data);

    % === 3. 当前 voxel 坐标 ===
    x0 = scan.x_axis(ix);
    y0 = scan.y_axis(iy);
    z0 = scan.z_axis(iz);

    % === 4. 当前 RC angle ===
    n_wave = angle_set(i_angle);
    angle_val = TX(n_wave).Steer(1);

    % === 5. RC 接收阵元位置 ===
    CRh_probe.x = ElementPos(129:256, 1);
    CRh_probe.y = ElementPos(129:256, 2);
    CRh_probe.z = ElementPos(129:256, 3);

    D = abs(CRh_probe.y(end) - CRh_probe.y(1));
    offset_distance = TW.peak * lambda;

    % === 6. 当前 voxel 的发射时间，对所有接收通道相同 ===
    transmit_distance = z0 * cos(angle_val) + ...
                        x0 * sin(angle_val) + ...
                        (D/2) * sin(angle_val) * sign(angle_val) + ...
                        offset_distance;

    transmit_time = transmit_distance / c0;

    % === 7. 对 128 个接收通道分别计算 receive delay ===
    n_channels = 128;

    receive_time = zeros(n_channels, 1);
    delay = zeros(n_channels, 1);
    sample_position = zeros(n_channels, 1);
    apo = zeros(n_channels, 1, 'single');

    sample = zeros(n_channels, 1);
    aligned_sample = zeros(n_channels, 1);

    time_vector = (0:(size(RF_data,1)-1)) / fs;

    rx_f_number = 1.5;

    for n_rx = 1:n_channels

        rx_y = CRh_probe.y(n_rx);
        rx_z = CRh_probe.z(n_rx);

        receive_distance = sqrt((rx_y - y0)^2 + (rx_z - z0)^2);
        receive_time(n_rx) = receive_distance / c0;

        delay(n_rx) = receive_time(n_rx) + transmit_time;

        sample_position(n_rx) = delay(n_rx) * fs + 1;

        sample(n_rx) = interp1(time_vector, RF_data(:, n_rx, i_angle), ...
                               delay(n_rx), 'linear', 0);

        apo(n_rx) = single(abs(rx_f_number * (rx_y - y0) / (rx_z - z0)) <= 0.5);

        aligned_sample(n_rx) = apo(n_rx) * sample(n_rx);
    end

    % === 8. 准备 RF 图像 ===
    rf_img = abs(RF_data(:, :, i_angle));  % [time, channel]

    % 只看 delay 附近的一段，方便观察
    win = 100;
    idx1 = max(floor(min(sample_position)) - win, 1);
    idx2 = min(ceil(max(sample_position)) + win, size(rf_img, 1));

    rf_local = rf_img(idx1:idx2, :);

    % dB 显示
    rf_db = 20 * log10(rf_local ./ max(rf_local(:)) + 1e-12);
    rf_db(rf_db < -60) = -60;

    sample_axis = idx1:idx2;

    % === 9. 画图：channel-time RF image + delay curve ===
    figure;
    imagesc(sample_axis, 1:n_channels, rf_db.');
    axis xy;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    hold on;

    plot(sample_position, 1:n_channels, 'r-', 'LineWidth', 1.8);
    plot(sample_position(apo == 1), find(apo == 1), 'ro', 'MarkerSize', 3, 'LineWidth', 1.0);
    plot(sample_position(apo == 0), find(apo == 0), 'bo', 'MarkerSize', 3, 'LineWidth', 1.0);

    xlabel('RF sample index');
    ylabel('Receive channel');
    title(sprintf('RC delay alignment over 128 channels: voxel [%d,%d,%d], angle %.1f deg', ...
          iz, ix, iy, angle_val * 180/pi));

    legend('delay curve', 'apo = 1', 'apo = 0');

    % === 10. 另一张图：128 个 delay-aligned samples ===
    figure;

    subplot(3,1,1);
    plot(1:n_channels, abs(sample), 'LineWidth', 1.2);
    grid on;
    xlabel('Receive channel');
    ylabel('|sample|');
    title('Interpolated RF samples before apodization');

    subplot(3,1,2);
    stem(1:n_channels, apo, 'filled');
    grid on;
    xlabel('Receive channel');
    ylabel('apo');
    title('Receive apodization mask');

    subplot(3,1,3);
    plot(1:n_channels, abs(aligned_sample), 'LineWidth', 1.2);
    grid on;
    xlabel('Receive channel');
    ylabel('|aligned sample|');
    title(sprintf('Delay-aligned samples after apodization, sum abs = %.3f', abs(sum(aligned_sample))));

    % === 11. 输出给后续验证用 ===
    out = struct();
    out.iz_ix_iy = [iz, ix, iy];
    out.x0_y0_z0_mm = [x0, y0, z0] * 1000;
    out.angle_deg = angle_val * 180/pi;
    out.transmit_time_us = transmit_time * 1e6;
    out.receive_time_us = receive_time * 1e6;
    out.delay_us = delay * 1e6;
    out.sample_position = sample_position;
    out.apodization = apo;
    out.sample = sample;
    out.aligned_sample = aligned_sample;
    out.RC_one_angle_sum = sum(aligned_sample);

    fprintf('\n=== RC one voxel, one angle, all channels ===\n');
    fprintf('Voxel index: iz=%d, ix=%d, iy=%d\n', iz, ix, iy);
    fprintf('Voxel position: x=%.3f mm, y=%.3f mm, z=%.3f mm\n', ...
            x0*1000, y0*1000, z0*1000);
    fprintf('Angle: %.3f deg\n', angle_val * 180/pi);
    fprintf('Transmit time: %.3f us\n', transmit_time * 1e6);
    fprintf('Sample position range: %.2f ~ %.2f\n', ...
            min(sample_position), max(sample_position));
    fprintf('Valid apodization channels: %d / 128\n', sum(apo == 1));
    fprintf('RC one-angle complex sum: %.4f + %.4fi\n', ...
            real(out.RC_one_angle_sum), imag(out.RC_one_angle_sum));
    fprintf('RC one-angle abs(sum): %.4f\n', abs(out.RC_one_angle_sum));
end

function out = debug_plot_RC_all_angles_one_voxel(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                                  angle_set, iz, ix, iy)

    % ============================================================
    % 目的：
    % 对一个 voxel，在 RC 分支下，计算所有 angle × 128 channels 的
    % delay-aligned RF samples。
    %
    % 输出：
    %   aligned_sample_mat: [128, N_angle]
    % ============================================================

    % === 1. 基本参数 ===
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;
    c0 = 1540;
    lambda = c0 / f0;
    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;
    N_angle = length(angle_set);
    N_ch = 128;

    % === 2. 提取 RC RF，与原 DAS 保持一致 ===
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                                'RC', angle_set, half_waves);
    RF_data = hilbert(RF_data);

    % === 3. voxel 坐标 ===
    x0 = scan.x_axis(ix);
    y0 = scan.y_axis(iy);
    z0 = scan.z_axis(iz);

    % === 4. RC 接收阵元位置 ===
    CRh_probe.x = ElementPos(129:256, 1);
    CRh_probe.y = ElementPos(129:256, 2);
    CRh_probe.z = ElementPos(129:256, 3);

    D = abs(CRh_probe.y(end) - CRh_probe.y(1));
    offset_distance = TW.peak * lambda;

    time_vector = (0:(size(RF_data,1)-1)) / fs;
    rx_f_number = 1.5;

    % === 5. 预分配 ===
    angle_deg = zeros(1, N_angle);
    transmit_time_us = zeros(1, N_angle);

    sample_position_mat = zeros(N_ch, N_angle);
    receive_time_us_mat = zeros(N_ch, N_angle);
    delay_us_mat = zeros(N_ch, N_angle);

    apo_mat = zeros(N_ch, N_angle, 'single');
    sample_mat = zeros(N_ch, N_angle);
    aligned_sample_mat = zeros(N_ch, N_angle);

    angle_sum = zeros(1, N_angle);

    % === 6. 主循环：angle × channel ===
    for ia = 1:N_angle

        n_wave = angle_set(ia);
        angle_val = TX(n_wave).Steer(1);
        angle_deg(ia) = angle_val * 180 / pi;

        % 当前角度下，发射时间对所有接收通道相同
        transmit_distance = z0 * cos(angle_val) + ...
                            x0 * sin(angle_val) + ...
                            (D/2) * sin(angle_val) * sign(angle_val) + ...
                            offset_distance;

        transmit_time = transmit_distance / c0;
        transmit_time_us(ia) = transmit_time * 1e6;

        for n_rx = 1:N_ch

            rx_y = CRh_probe.y(n_rx);
            rx_z = CRh_probe.z(n_rx);

            receive_distance = sqrt((rx_y - y0)^2 + (rx_z - z0)^2);
            receive_time = receive_distance / c0;

            delay = receive_time + transmit_time;

            sample_position = delay * fs + 1;

            sample = interp1(time_vector, RF_data(:, n_rx, ia), ...
                             delay, 'linear', 0);

            apo = single(abs(rx_f_number * (rx_y - y0) / (rx_z - z0)) <= 0.5);

            aligned_sample = apo * sample;

            receive_time_us_mat(n_rx, ia) = receive_time * 1e6;
            delay_us_mat(n_rx, ia) = delay * 1e6;
            sample_position_mat(n_rx, ia) = sample_position;

            apo_mat(n_rx, ia) = apo;
            sample_mat(n_rx, ia) = sample;
            aligned_sample_mat(n_rx, ia) = aligned_sample;
        end

        angle_sum(ia) = sum(aligned_sample_mat(:, ia));
    end

    RC_total_sum = sum(angle_sum);

    % ============================================================
    % 图 1：|aligned sample| 的 channel-angle 图
    % ============================================================
    img1 = abs(aligned_sample_mat);

    figure;
    imagesc(1:N_angle, 1:N_ch, img1);
    axis xy;
    colorbar;
    xlabel('Angle index in angle\_set');
    ylabel('Receive channel');
    title(sprintf('RC aligned samples |F(ch,angle)|, voxel [%d,%d,%d]', iz, ix, iy));
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    % ============================================================
    % 图 2：apodization mask
    % ============================================================
    figure;
    imagesc(1:N_angle, 1:N_ch, apo_mat);
    axis xy;
    colorbar;
    xlabel('Angle index in angle\_set');
    ylabel('Receive channel');
    title('RC apodization mask');
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    % ============================================================
    % 图 3：每个角度的求和贡献
    % ============================================================
    figure;
    subplot(2,1,1);
    stem(1:N_angle, abs(angle_sum), 'filled');
    grid on;
    xlabel('Angle index in angle\_set');
    ylabel('|sum over channels|');
    title('Per-angle RC contribution');
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    subplot(2,1,2);
    plot(1:N_angle, real(angle_sum), '-o', 'LineWidth', 1.2); hold on;
    plot(1:N_angle, imag(angle_sum), '-s', 'LineWidth', 1.2);
    grid on;
    xlabel('Angle index in angle\_set');
    ylabel('Complex sum');
    title('Real / Imag of per-angle RC contribution');
    legend('Real', 'Imag');
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    % ============================================================
    % 输出
    % ============================================================
    out = struct();
    out.voxel_index = [iz, ix, iy];
    out.voxel_mm = [x0, y0, z0] * 1000;

    out.angle_set = angle_set;
    out.angle_deg = angle_deg;
    out.transmit_time_us = transmit_time_us;

    out.sample_position_mat = sample_position_mat;
    out.receive_time_us_mat = receive_time_us_mat;
    out.delay_us_mat = delay_us_mat;

    out.apo_mat = apo_mat;
    out.sample_mat = sample_mat;
    out.aligned_sample_mat = aligned_sample_mat;

    out.angle_sum = angle_sum;
    out.RC_total_sum = RC_total_sum;

    % 命令行摘要
    fprintf('\n=== RC one voxel, all angles, all channels ===\n');
    fprintf('Voxel index: iz=%d, ix=%d, iy=%d\n', iz, ix, iy);
    fprintf('Voxel position: x=%.3f mm, y=%.3f mm, z=%.3f mm\n', ...
            x0*1000, y0*1000, z0*1000);

    for ia = 1:N_angle
        fprintf('\n[Angle %d / %d]\n', ia, N_angle);
        fprintf('  n_wave = %d\n', angle_set(ia));
        fprintf('  angle = %.3f deg\n', angle_deg(ia));
        fprintf('  transmit time = %.3f us\n', transmit_time_us(ia));
        fprintf('  sample position range = %.2f ~ %.2f\n', ...
                min(sample_position_mat(:, ia)), max(sample_position_mat(:, ia)));
        fprintf('  valid apo channels = %d / 128\n', sum(apo_mat(:, ia) == 1));
        fprintf('  angle sum = %.4f + %.4fi\n', ...
                real(angle_sum(ia)), imag(angle_sum(ia)));
        fprintf('  abs(angle sum) = %.4f\n', abs(angle_sum(ia)));
    end

    fprintf('\nRC total sum over all angles = %.4f + %.4fi\n', ...
            real(RC_total_sum), imag(RC_total_sum));
    fprintf('RC total abs(sum) = %.4f\n', abs(RC_total_sum));
end

function out = debug_plot_CR_all_angles_one_voxel(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                                  angle_set, iz, ix, iy)

    % ============================================================
    % 目的：
    % 对一个 voxel，在 CR 分支下，计算所有 angle × 128 channels 的
    % delay-aligned RF samples。
    %
    % 输出：
    %   aligned_sample_mat: [128, N_angle]
    % ============================================================

    % === 1. 基本参数 ===
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;
    c0 = 1540;
    lambda = c0 / f0;
    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;
    N_angle = length(angle_set);
    N_ch = 128;

    % === 2. 提取 CR RF，与原 DAS 保持一致 ===
    RF_data = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                                'CR', angle_set, half_waves);
    RF_data = hilbert(RF_data);

    % === 3. voxel 坐标 ===
    x0 = scan.x_axis(ix);
    y0 = scan.y_axis(iy);
    z0 = scan.z_axis(iz);

    % === 4. CR 接收阵元位置 ===
    RRh_probe.x = ElementPos(1:128, 1);
    RRh_probe.y = ElementPos(1:128, 2);
    RRh_probe.z = ElementPos(1:128, 3);

    D = abs(RRh_probe.x(end) - RRh_probe.x(1));
    offset_distance = TW.peak * lambda;

    time_vector = (0:(size(RF_data,1)-1)) / fs;
    rx_f_number = 1.5;

    % === 5. 预分配 ===
    angle_deg = zeros(1, N_angle);
    transmit_time_us = zeros(1, N_angle);

    sample_position_mat = zeros(N_ch, N_angle);
    receive_time_us_mat = zeros(N_ch, N_angle);
    delay_us_mat = zeros(N_ch, N_angle);

    apo_mat = zeros(N_ch, N_angle, 'single');
    sample_mat = zeros(N_ch, N_angle);
    aligned_sample_mat = zeros(N_ch, N_angle);

    angle_sum = zeros(1, N_angle);

    % === 6. 主循环：angle × channel ===
    for ia = 1:N_angle

        n_wave_local = angle_set(ia);

        % CR 的角度来自后一半 TX，并取 Steer(2)
        n_wave_global = half_waves + n_wave_local;
        angle_val = TX(n_wave_global).Steer(2);

        angle_deg(ia) = angle_val * 180 / pi;

        % 当前角度下，CR 发射时间对所有接收通道相同
        % 注意：CR 发射方向看 y-z 平面，所以这里是 y0 * sin(angle)
        transmit_distance = z0 * cos(angle_val) + ...
                            y0 * sin(angle_val) + ...
                            (D/2) * sin(angle_val) * sign(angle_val) + ...
                            offset_distance;

        transmit_time = transmit_distance / c0;
        transmit_time_us(ia) = transmit_time * 1e6;

        for n_rx = 1:N_ch

            rx_x = RRh_probe.x(n_rx);
            rx_z = RRh_probe.z(n_rx);

            % CR 接收距离看 x-z 平面
            receive_distance = sqrt((rx_x - x0)^2 + (rx_z - z0)^2);
            receive_time = receive_distance / c0;

            delay = receive_time + transmit_time;

            sample_position = delay * fs + 1;

            sample = interp1(time_vector, RF_data(:, n_rx, ia), ...
                             delay, 'linear', 0);

            % CR 的接收孔径也在 x-z 平面
            apo = single(abs(rx_f_number * (rx_x - x0) / (rx_z - z0)) <= 0.5);

            aligned_sample = apo * sample;

            receive_time_us_mat(n_rx, ia) = receive_time * 1e6;
            delay_us_mat(n_rx, ia) = delay * 1e6;
            sample_position_mat(n_rx, ia) = sample_position;

            apo_mat(n_rx, ia) = apo;
            sample_mat(n_rx, ia) = sample;
            aligned_sample_mat(n_rx, ia) = aligned_sample;
        end

        angle_sum(ia) = sum(aligned_sample_mat(:, ia));
    end

    CR_total_sum = sum(angle_sum);

    % ============================================================
    % 图 1：|aligned sample| 的 channel-angle 图
    % ============================================================
    img1 = abs(aligned_sample_mat);

    figure;
    imagesc(1:N_angle, 1:N_ch, img1);
    axis xy;
    colorbar;
    xlabel('Angle index in angle\_set');
    ylabel('Receive channel');
    title(sprintf('CR aligned samples |F(ch,angle)|, voxel [%d,%d,%d]', iz, ix, iy));
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    % ============================================================
    % 图 2：apodization mask
    % ============================================================
    figure;
    imagesc(1:N_angle, 1:N_ch, apo_mat);
    axis xy;
    colorbar;
    xlabel('Angle index in angle\_set');
    ylabel('Receive channel');
    title('CR apodization mask');
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    % ============================================================
    % 图 3：每个角度的求和贡献
    % ============================================================
    figure;
    subplot(2,1,1);
    stem(1:N_angle, abs(angle_sum), 'filled');
    grid on;
    xlabel('Angle index in angle\_set');
    ylabel('|sum over channels|');
    title('Per-angle CR contribution');
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    subplot(2,1,2);
    plot(1:N_angle, real(angle_sum), '-o', 'LineWidth', 1.2); hold on;
    plot(1:N_angle, imag(angle_sum), '-s', 'LineWidth', 1.2);
    grid on;
    xlabel('Angle index in angle\_set');
    ylabel('Complex sum');
    title('Real / Imag of per-angle CR contribution');
    legend('Real', 'Imag');
    xticks(1:N_angle);
    xticklabels(compose('%.1f°', angle_deg));

    % ============================================================
    % 输出
    % ============================================================
    out = struct();
    out.voxel_index = [iz, ix, iy];
    out.voxel_mm = [x0, y0, z0] * 1000;

    out.angle_set = angle_set;
    out.angle_deg = angle_deg;
    out.transmit_time_us = transmit_time_us;

    out.sample_position_mat = sample_position_mat;
    out.receive_time_us_mat = receive_time_us_mat;
    out.delay_us_mat = delay_us_mat;

    out.apo_mat = apo_mat;
    out.sample_mat = sample_mat;
    out.aligned_sample_mat = aligned_sample_mat;

    out.angle_sum = angle_sum;
    out.CR_total_sum = CR_total_sum;

    % 命令行摘要
    fprintf('\n=== CR one voxel, all angles, all channels ===\n');
    fprintf('Voxel index: iz=%d, ix=%d, iy=%d\n', iz, ix, iy);
    fprintf('Voxel position: x=%.3f mm, y=%.3f mm, z=%.3f mm\n', ...
            x0*1000, y0*1000, z0*1000);

    for ia = 1:N_angle
        fprintf('\n[Angle %d / %d]\n', ia, N_angle);
        fprintf('  n_wave_local = %d\n', angle_set(ia));
        fprintf('  n_wave_global = %d\n', half_waves + angle_set(ia));
        fprintf('  angle = %.3f deg\n', angle_deg(ia));
        fprintf('  transmit time = %.3f us\n', transmit_time_us(ia));
        fprintf('  sample position range = %.2f ~ %.2f\n', ...
                min(sample_position_mat(:, ia)), max(sample_position_mat(:, ia)));
        fprintf('  valid apo channels = %d / 128\n', sum(apo_mat(:, ia) == 1));
        fprintf('  angle sum = %.4f + %.4fi\n', ...
                real(angle_sum(ia)), imag(angle_sum(ia)));
        fprintf('  abs(angle sum) = %.4f\n', abs(angle_sum(ia)));
    end

    fprintf('\nCR total sum over all angles = %.4f + %.4fi\n', ...
            real(CR_total_sum), imag(CR_total_sum));
    fprintf('CR total abs(sum) = %.4f\n', abs(CR_total_sum));
end

function out = extract_aligned_patch_RC_CR_debug(RcvData, Trans, Resource, TX, TW, Receive, scan, ...
                                                 angle_set, z_idx, x_idx, y_idx)

    % ============================================================
    % 目的：
    % 生成一个小 patch 的 delay-aligned RF tensor。
    %
    % 输出：
    %   F_RC_patch: [Nz_patch, Nx_patch, Ny_patch, 128, N_angle]
    %   F_CR_patch: [Nz_patch, Nx_patch, Ny_patch, 128, N_angle]
    %
    % 验证：
    %   sum(F_RC_patch, channel, angle) + sum(F_CR_patch, channel, angle)
    %   ≈
    %   reconstruct_RC_Phatom(scan_patch) + reconstruct_CR_Phatom(scan_patch)
    % ============================================================

    % === 1. 构造 patch scan，必须沿用原 scan 的坐标轴 ===
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

    % === 2. 基本参数 ===
    f0 = double(Trans.frequency * 1e6);
    fs = f0 * Receive(1).samplesPerWave;
    c0 = 1540;
    lambda = c0 / f0;
    ElementPos = Trans.ElementPos .* lambda;

    half_waves = length(TX) / 2;
    time_vector = (0:(Receive(1).endSample-1)) / fs;

    rx_f_number = 1.5;
    offset_distance = TW.peak * lambda;

    % ============================================================
    % 3. 提取 RC / CR RF 数据，并 Hilbert
    % ============================================================

    RF_RC = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                              'RC', angle_set, half_waves);
    RF_RC = hilbert(RF_RC);

    RF_CR = extract_rf_subset(RcvData, Receive, Resource, Trans, ...
                              'CR', angle_set, half_waves);
    RF_CR = hilbert(RF_CR);

    % ============================================================
    % 4. 预分配 delay-aligned tensor
    % ============================================================

    F_RC_patch = complex(zeros(Nz, Nx, Ny, N_ch, N_angle, 'single'));
    F_CR_patch = complex(zeros(Nz, Nx, Ny, N_ch, N_angle, 'single'));

    % 为了 debug，也保存 delay sample position 的范围
    sample_pos_RC_minmax = zeros(N_angle, 2);
    sample_pos_CR_minmax = zeros(N_angle, 2);

    apo_count_RC = zeros(N_angle, 1);
    apo_count_CR = zeros(N_angle, 1);

    % ============================================================
    % 5. RC patch aligned tensor
    % ============================================================

    % RC: 发射角度用 TX(1:half).Steer(1)
    alpha = zeros(half_waves, 1);
    for k = 1:half_waves
        alpha(k) = TX(k).Steer(1);
    end

    % RC: 接收阵元用 ElementPos(129:256)，接收距离看 y-z 平面
    CRh_probe.x = ElementPos(129:256, 1);
    CRh_probe.y = ElementPos(129:256, 2);
    CRh_probe.z = ElementPos(129:256, 3);

    Cym = CRh_probe.y.' - scan_patch.y(:);
    Czm = CRh_probe.z.' - scan_patch.z(:);

    receive_delay_RC = single(sqrt(Cym.^2 + Czm.^2) / c0);
    apo_RC = single(abs(rx_f_number .* Cym ./ Czm) <= 0.5);

    D_RC = abs(CRh_probe.y(end) - CRh_probe.y(1));

    fprintf('\n=== Extract RC aligned patch ===\n');

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
                   interp1(time_vector, RF_RC(:, n_rx, ia), double(delay), 'linear', 0);

            F_RC_patch(:, :, :, n_rx, ia) = reshape(temp, [Nz, Nx, Ny]);
        end

        sample_pos_RC_minmax(ia, :) = [min(all_sample_pos(:)), max(all_sample_pos(:))];
        apo_count_RC(ia) = sum(apo_RC(:) == 1) / Nvox;

        fprintf('[RC angle %d/%d] angle = %.3f deg, sample position range = %.2f ~ %.2f, mean valid apo channels = %.1f / 128\n', ...
            ia, N_angle, angle_val * 180/pi, ...
            sample_pos_RC_minmax(ia,1), sample_pos_RC_minmax(ia,2), apo_count_RC(ia));
    end

    % ============================================================
    % 6. CR patch aligned tensor
    % ============================================================

    % CR: 发射角度用 TX(half+1:end).Steer(2)
    beta = zeros(half_waves, 1);
    for k = 1:half_waves
        beta(k) = TX(half_waves + k).Steer(2);
    end

    % CR: 接收阵元用 ElementPos(1:128)，接收距离看 x-z 平面
    RRh_probe.x = ElementPos(1:128, 1);
    RRh_probe.y = ElementPos(1:128, 2);
    RRh_probe.z = ElementPos(1:128, 3);

    Rxm = RRh_probe.x.' - scan_patch.x(:);
    Rzm = RRh_probe.z.' - scan_patch.z(:);

    receive_delay_CR = single(sqrt(Rxm.^2 + Rzm.^2) / c0);
    apo_CR = single(abs(rx_f_number .* Rxm ./ Rzm) <= 0.5);

    D_CR = abs(RRh_probe.x(end) - RRh_probe.x(1));

    fprintf('\n=== Extract CR aligned patch ===\n');

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
                   interp1(time_vector, RF_CR(:, n_rx, ia), double(delay), 'linear', 0);

            F_CR_patch(:, :, :, n_rx, ia) = reshape(temp, [Nz, Nx, Ny]);
        end

        sample_pos_CR_minmax(ia, :) = [min(all_sample_pos(:)), max(all_sample_pos(:))];
        apo_count_CR(ia) = sum(apo_CR(:) == 1) / Nvox;

        fprintf('[CR angle %d/%d] angle = %.3f deg, sample position range = %.2f ~ %.2f, mean valid apo channels = %.1f / 128\n', ...
            ia, N_angle, angle_val * 180/pi, ...
            sample_pos_CR_minmax(ia,1), sample_pos_CR_minmax(ia,2), apo_count_CR(ia));
    end

    % ============================================================
    % 7. tensor 求和，得到 patch DAS
    % ============================================================

    tmp_RC = sum(sum(F_RC_patch, 5), 4);
    tmp_CR = sum(sum(F_CR_patch, 5), 4);

    vol_RC_from_tensor = reshape(tmp_RC, [Nz, Nx, Ny]);
    vol_CR_from_tensor = reshape(tmp_CR, [Nz, Nx, Ny]);

    vol_from_tensor = vol_RC_from_tensor + vol_CR_from_tensor;

    % ============================================================
    % 8. 用原始 DAS 函数重建同一个 patch
    % ============================================================

    fprintf('\n=== Original DAS patch reconstruction ===\n');

    vol_RC_original = reconstruct_RC_Phatom( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan_patch, angle_set);

    vol_CR_original = reconstruct_CR_Phatom( ...
        RcvData, Trans, Resource, TX, TW, Receive, scan_patch, angle_set);

    vol_original = vol_RC_original + vol_CR_original;

    % ============================================================
    % 9. 误差统计
    % ============================================================

    err_patch = vol_original - vol_from_tensor;

    abs_diff = abs(err_patch);
    abs_original = abs(vol_original);

    max_abs_diff = max(abs_diff(:));
    mean_abs_diff = mean(abs_diff(:));

    rel_l2_diff = norm(err_patch(:)) / (norm(vol_original(:)) + eps);
    max_rel_diff = max(abs_diff(:) ./ (abs_original(:) + eps));

    fprintf('\n=== Patch validation result ===\n');
    fprintf('Patch size: Nz × Nx × Ny = %d × %d × %d\n', Nz, Nx, Ny);
    fprintf('Angles: %d\n', N_angle);
    fprintf('F_RC_patch size: [%d %d %d %d %d]\n', size(F_RC_patch));
    fprintf('F_CR_patch size: [%d %d %d %d %d]\n', size(F_CR_patch));
    fprintf('Max abs difference  = %.6e\n', max_abs_diff);
    fprintf('Mean abs difference = %.6e\n', mean_abs_diff);
    fprintf('Relative L2 diff    = %.6e\n', rel_l2_diff);
    fprintf('Max relative diff   = %.6e\n', max_rel_diff);

    % 中心体素再单独打印一次
    izc = ceil(Nz/2);
    ixc = ceil(Nx/2);
    iyc = ceil(Ny/2);

    fprintf('\nCenter voxel in patch: local [%d,%d,%d], global [%d,%d,%d]\n', ...
        izc, ixc, iyc, z_idx(izc), x_idx(ixc), y_idx(iyc));

    fprintf('Original center voxel = %.6f + %.6fi\n', ...
        real(vol_original(izc,ixc,iyc)), imag(vol_original(izc,ixc,iyc)));

    fprintf('Tensor center voxel   = %.6f + %.6fi\n', ...
        real(vol_from_tensor(izc,ixc,iyc)), imag(vol_from_tensor(izc,ixc,iyc)));

    fprintf('Center abs diff       = %.6e\n', ...
        abs(vol_original(izc,ixc,iyc) - vol_from_tensor(izc,ixc,iyc)));

    % ============================================================
    % 10. 输出
    % ============================================================

    out = struct();

    out.z_idx = z_idx;
    out.x_idx = x_idx;
    out.y_idx = y_idx;
    out.scan_patch = scan_patch;
    out.angle_set = angle_set;

    out.F_RC_patch = F_RC_patch;
    out.F_CR_patch = F_CR_patch;

    out.vol_RC_from_tensor = vol_RC_from_tensor;
    out.vol_CR_from_tensor = vol_CR_from_tensor;
    out.vol_from_tensor = vol_from_tensor;

    out.vol_RC_original = vol_RC_original;
    out.vol_CR_original = vol_CR_original;
    out.vol_original = vol_original;

    out.err_patch = err_patch;

    out.max_abs_diff = max_abs_diff;
    out.mean_abs_diff = mean_abs_diff;
    out.rel_l2_diff = rel_l2_diff;
    out.max_rel_diff = max_rel_diff;

    out.sample_pos_RC_minmax = sample_pos_RC_minmax;
    out.sample_pos_CR_minmax = sample_pos_CR_minmax;
    out.apo_count_RC = apo_count_RC;
    out.apo_count_CR = apo_count_CR;
end

function plot_patch_validation_debug(out)

    scan_patch = out.scan_patch;

    vol_original = out.vol_original;
    vol_from_tensor = out.vol_from_tensor;
    err_patch = out.err_patch;

    Nz = scan_patch.N_z;
    Nx = scan_patch.N_x;
    Ny = scan_patch.N_y;

    % 选 patch 中间的 y slice
    iy_mid = ceil(Ny / 2);

    x_mm = scan_patch.x_axis * 1000;
    z_mm = scan_patch.z_axis * 1000;

    img_original = abs(squeeze(vol_original(:, :, iy_mid)));
    img_tensor   = abs(squeeze(vol_from_tensor(:, :, iy_mid)));
    img_error    = abs(squeeze(err_patch(:, :, iy_mid)));

    % 为了显示方便，共用一个幅值范围
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
    title('Sum of aligned RF tensor');

    subplot(1,3,3);
    imagesc(x_mm, z_mm, img_error_db);
    axis image;
    colormap gray;
    colorbar;
    caxis([-60 0]);
    xlabel('x [mm]');
    ylabel('z [mm]');
    title('|Difference|');

    sgtitle(sprintf('Patch validation, y-slice %d / %d', iy_mid, Ny));

    % 再画一个绝对误差分布
    figure;
    histogram(abs(err_patch(:)), 50);
    grid on;
    xlabel('|Original - Tensor sum|');
    ylabel('Count');
    title(sprintf('Patch error histogram, Relative L2 = %.3e', out.rel_l2_diff));
end