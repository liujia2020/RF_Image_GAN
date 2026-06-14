clc; clear; close all;

% =========================================================================

% === [1. 配置区域] ===

% =========================================================================

InputRoot  = 'G:\Data_0110_RFdata\Phantom_Data';       % 输入根目录
OutputRoot = 'G:\DAS\Phantom_Data\02_DAS_Result';    % 输出根目录

% 任务定义
idx_SQ = 1:75;
idx_LQ = [3, 38, 73]; 

Tasks = struct('name', {}, 'indices', {}, 'folder', {});
cnt = 0;
% 任务 1: SQ (75角)
cnt=cnt+1; Tasks(cnt).name='SQ'; Tasks(cnt).indices=idx_SQ; Tasks(cnt).folder='Recon_SQ_75';
% 任务 2: LQ (3角)
cnt=cnt+1; Tasks(cnt).name='LQ'; Tasks(cnt).indices=idx_LQ; Tasks(cnt).folder='Recon_LQ_03';

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