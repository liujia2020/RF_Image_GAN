clc; clear; close all;

% =========================================================================
% === [0. 突破 MATLAB 限制，强制指定多核] ===
% =========================================================================
targetWorkers = 10; % 强制使用 32 个核心

poolobj = gcp('nocreate');
if isempty(poolobj)
    fprintf('🔥 正在启动并行池 (强制指定核心数: %d)...\n', targetWorkers);
    parpool('local', targetWorkers);
elseif poolobj.NumWorkers ~= targetWorkers
    fprintf('♻️ 当前池子核心数只有 %d，正在强制重启为 %d 核...\n', poolobj.NumWorkers, targetWorkers);
    delete(poolobj);
    parpool('local', targetWorkers);
else
    fprintf('✅ 已成功连接到 %d 核并行池\n', poolobj.NumWorkers);
end

% =========================================================================
% === [1. 配置区域] ===
% =========================================================================
InputRoot  = 'G:\Data_0110_RFdata\Carotid_Data\02_RF_Data\';       
OutputRoot = 'G:\Data_0110_RFdata\Carotid_Data\03_DAS_Result\';  

% --- 开关: 控制本次跑哪些 (1=跑, 0=不跑) ---
DO_SQ = 0;   % SQ (75角, Ground Truth)
DO_HQ = 0;   % HQ (33角, 对照)
DO_LQ = 1;   % LQ (3角,  稀疏输入)

% --- 定义不同质量的重建角度索引 ---
idx_SQ = 1:75;
idx_HQ = [1, 3, 6, 8, 10, 13, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, ...
          38, ... 
          40, 43, 45, 47, 50, 52, 54, 57, 59, 61, 63, 66, 68, 70, 73, 75];
idx_LQ = [3, 38, 73]; 

% --- 动态装载到并行任务队列 ---
Tasks = struct('name', {}, 'indices', {}, 'folder', {});
taskIdx = 1; % 任务计数器

if DO_SQ == 1
    Tasks(taskIdx).name = 'SQ';                   
    Tasks(taskIdx).indices = idx_SQ;              
    Tasks(taskIdx).folder = 'Recon_SQ_75';        
    taskIdx = taskIdx + 1;
end

if DO_HQ == 1
    Tasks(taskIdx).name = 'HQ';                   
    Tasks(taskIdx).indices = idx_HQ;              
    Tasks(taskIdx).folder = 'Recon_HQ_33';        
    taskIdx = taskIdx + 1;
end

if DO_LQ == 1
    Tasks(taskIdx).name = 'LQ';                   
    Tasks(taskIdx).indices = idx_LQ;              
    Tasks(taskIdx).folder = 'Recon_LQ_03';        
    taskIdx = taskIdx + 1; % 这行虽然是最后，但养成习惯加上，方便以后扩展
end

% 安全检查：防止你全关了导致后面代码报错
if isempty(Tasks)
    error('🚨 致命错误：所有重建开关都被设置为 0 了！请至少打开一个任务。');
end

if ~exist(OutputRoot, 'dir'), mkdir(OutputRoot); end
fprintf('📂 正在预创建输出目录 (本次共激活 %d 个重建任务)...\n', length(Tasks));
for t = 1:length(Tasks)
    TaskRoot = fullfile(OutputRoot, Tasks(t).folder);
    if ~exist(fullfile(TaskRoot, 'MAT'), 'dir'), mkdir(fullfile(TaskRoot, 'MAT')); end
    if ~exist(fullfile(TaskRoot, 'NII'), 'dir'), mkdir(fullfile(TaskRoot, 'NII')); end
    if ~exist(fullfile(TaskRoot, 'PNG'), 'dir'), mkdir(fullfile(TaskRoot, 'PNG')); end
end

% =========================================================================
% === [2. 解决硬盘风暴：全内存预加载 (RAM Cache) - 必须要有！] ===
% =========================================================================
fileList = dir(fullfile(InputRoot, '**', '*.mat'));
fileList = fileList(~startsWith({fileList.name}, '.')); 
N_Files = length(fileList);

fprintf('======================================================\n');
fprintf('   🚀 阶段 1/2: 正在将硬盘数据吸入内存...\n');
fprintf('   💾 待加载文件数: %d (请耐心等待几十秒)\n', N_Files);
fprintf('======================================================\n');

DataCache = cell(1, N_Files);
for i = 1:N_Files
    fullPath = fullfile(fileList(i).folder, fileList(i).name);
    raw = load(fullPath); % 串行单线程读取，硬盘最舒服的方式
    
    CacheItem = struct();
    CacheItem.filename = fileList(i).name;
    CacheItem.Resource = raw.Resource;
    if iscell(raw.RcvData), rcv=raw.RcvData{1}; else, rcv=raw.RcvData; end
    CacheItem.RF_Single = rcv(:, :, 1); 
    CacheItem.Receive = raw.Receive(1:length(raw.TX));
    CacheItem.Trans = raw.Trans;
    CacheItem.TX = raw.TX;
    CacheItem.TW = raw.TW;
    DataCache{i} = CacheItem;
    
    if mod(i, 10) == 0 || i == N_Files
        fprintf('   ✅ 已加载至内存: %d / %d\n', i, N_Files);
    end
end
fprintf('\n🎉 内存预加载完毕！启动多核矩阵飙车！\n\n');

% [关键]: 使用 DataQueue 接收并行 worker 传来的实时进度
q = parallel.pool.DataQueue;
afterEach(q, @(msg) fprintf('%s\n', msg));

%% =========================================================================
% === [3. 阶段 2/2: 并行循环 (0 IO 延迟计算)] ===
% =========================================================================
parfor i = 1 : N_Files
    
    % --- 直接从内存 Cache 中提取数据，绝对不去读硬盘！ ---
    D = DataCache{i};
    [~, baseName, ~] = fileparts(D.filename);
    
    send(q, sprintf(' -> [开始计算] 核心接管: %s', baseName));
    
    Res = D.Resource; 
    Res.RcvBuffer(1).numFrames = 1; 
    RF_Single = D.RF_Single;
    Rec_S = D.Receive; 
    Trans = D.Trans; 
    TX = D.TX; 
    TW = D.TW;
    
    % --- 准备网格 ---
    pitch = 0.2e-3;
    scan = struct();
    scan.startdepth = 5e-3; scan.enddepth = 42e-3; 
    scan.N_z = 1024; scan.N_x = 128; scan.N_y = 128; 
    
    scan.x_axis = linspace(-127*pitch/2, 127*pitch/2, scan.N_x);
    scan.y_axis = linspace(-127*pitch/2, 127*pitch/2, scan.N_y);
    scan.z_axis = linspace(scan.startdepth, scan.enddepth, scan.N_z);
    
    for t = 1:length(Tasks)
        currTask = Tasks(t);
        TaskRoot = fullfile(OutputRoot, currTask.folder);
        FileTag = sprintf('%s_%s', baseName, lower(currTask.name));
        
        MatPath = fullfile(TaskRoot, 'MAT', [FileTag, '.mat']);
        NiiPath = fullfile(TaskRoot, 'NII', [FileTag, '.nii']);
        PngPath = fullfile(TaskRoot, 'PNG', [FileTag, '_Views.png']);
        
        if exist(MatPath, 'file')
            continue; % 跳过已计算的
        end
        
        % 1. 重建 RC (矩阵极速版)
        vol_RC = reconstruct_RC_Sliced(RF_Single, Trans, Res, TX, TW, Rec_S, scan, currTask.indices, q, baseName);
        
        % 2. 重建 CR (矩阵极速版)
        vol_CR = reconstruct_CR_Sliced(RF_Single, Trans, Res, TX, TW, Rec_S, scan, currTask.indices, q, baseName);
        
        % 3. 融合
        volume_final = vol_RC + vol_CR; 
        
        % 4. 保存 MAT
        par_save_mat(MatPath, volume_final, scan.x_axis, scan.y_axis, scan.z_axis);
        

        % 5. 保存 NIfTI
        vol_abs = abs(volume_final);
        max_val = max(vol_abs(:)); if max_val==0, max_val=1; end
        
        % 这里的 log 压缩顺便帮你把动态范围规范化了
        vol_db = 20 * log10(vol_abs ./ max_val + 1e-12);
        vol_db(vol_db < -60) = -60; vol_db(vol_db > 0) = 0;
        vol_nii = single(vol_db); 
        
        dx = abs(scan.x_axis(2)-scan.x_axis(1))*1000; % 0.2
        dy = abs(scan.y_axis(2)-scan.y_axis(1))*1000; % 0.2
        dz = abs(scan.z_axis(2)-scan.z_axis(1))*1000; % 大约 0.0362
        vox_size = [dz, dx, dy]; 
        
        % --- 【完美解决 MATLAB NIfTI 无法覆盖保存的 Bug】 ---
        if exist(NiiPath, 'file')
            delete(NiiPath); % 确保没有旧文件拦截
        end
        % 1. 先生成一个基础版本
        niftiwrite(vol_nii, NiiPath, 'Compressed', false);
        
        % 2. 读出它的信息壳子
        info = niftiinfo(NiiPath);
        info.PixelDimensions = vox_size;
        info.SpaceUnits = 'Millimeter';
        
        % 3. 必须删掉刚刚生成的原文件，否则写入会报错被拦截！
        delete(NiiPath); 
        
        % 4. 重新注入灵魂（包含正确维度的头文件）
        niftiwrite(vol_nii, NiiPath, info, 'Compressed', false);
        
%         % 5. 保存 NIfTI
%         vol_abs = abs(volume_final);
%         max_val = max(vol_abs(:)); if max_val==0, max_val=1; end
%         vol_db = 20 * log10(vol_abs ./ max_val + 1e-12);
%         vol_db(vol_db < -60) = -60; vol_db(vol_db > 0) = 0;
%         vol_nii = single(vol_db); 
%         
%         dx = abs(scan.x_axis(2)-scan.x_axis(1))*1000; 
%         dy = abs(scan.y_axis(2)-scan.y_axis(1))*1000;
%         dz = abs(scan.z_axis(2)-scan.z_axis(1))*1000;
%         vox_size = [dz, dx, dy]; 
%         
%         niftiwrite(vol_nii, NiiPath, 'Compressed', false);
%         try
%             info = niftiinfo(NiiPath);
%             info.PixelDimensions = vox_size;
%             info.SpaceUnits = 'Millimeter';
%             niftiwrite(vol_nii, NiiPath, info, 'Compressed', false);
%         catch
%         end
        

        % 6. 保存三视图
        save_ortho_views(volume_final, scan, PngPath, FileTag);
    end
    
    send(q, sprintf(' ✅✅ [大功告成] 文件进度: %d / %d (%s)', i, N_Files, baseName));
end
fprintf('\n🎉 全部并行处理完成！\n');

% =========================================================================
% === 以下全为局部函数，必须放在文件最末尾，绝不能有其他主程序代码 ===
% =========================================================================

function par_save_mat(fname, volume_final, x, y, z)
    save(fname, 'volume_final', 'x', 'y', 'z', '-v7.3');
end

% --- [底层核心] 1. RC 重建 (极致向量化，消灭 interp1) ---
function vol_RC = reconstruct_RC_Sliced(RcvData, Trans, Resource, TX, TW, Receive, scan, angle_set, q, baseName)
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
    [nS, nCh, ~] = size(RF_data); 
    
    vol_RC = zeros(scan.N_z, scan.N_x, scan.N_y, 'single'); 
    
    ProbeY = ElementPos(129:256,2).'; 
    ProbeZ = ElementPos(129:256,3).'; 
    D = abs(ProbeY(end)-ProbeY(1));
    offset_distance = TW.peak*lambda;
    
    z_ax = scan.z_axis(:); x_ax = scan.x_axis(:); y_ax = scan.y_axis(:);
    
    for iy = 1:scan.N_y
        % --- 【切片级实时心跳包】 ---
        if mod(iy, 16) == 0 || iy == 1
            send(q, sprintf('      ⏳ [%s] RC 进度: 切片 %d / %d', baseName, iy, scan.N_y));
        end
        
        y_val = y_ax(iy);
        [grid_x, grid_z] = meshgrid(x_ax, z_ax);
        flat_x = grid_x(:); flat_z = grid_z(:);
        
        Pz_rel = flat_z - ProbeZ; 
        Py_rel = y_val - ProbeY; 
        RxDist = sqrt(Py_rel.^2 + Pz_rel.^2);
        
        rx_f_num = 1.5;
        ApoMask = abs(rx_f_num .* Py_rel ./ Pz_rel) <= 0.5;
        
        slice_accum = zeros(length(flat_x), 1, 'single');
        
        for i_ang = 1:length(angle_set)
            n_wave = angle_set(i_ang);
            angle_val = alpha(n_wave);
            
            TxDelay = flat_z*cos(angle_val) + flat_x*sin(angle_val) + ...
                      (D/2)*sin(angle_val)*sign(angle_val) + offset_distance;
            
            % --- 纯矩阵运算代替 interp1 ---
            Tidx = (RxDist + TxDelay) .* (fs/c0) + 1; 
            
            idx_fl = floor(Tidx);
            frac = Tidx - idx_fl;
            
            mask = (idx_fl >= 1) & (idx_fl < nS);
            idx_fl = max(1, min(nS-1, idx_fl));
            
            ch_off = (0:nCh-1) .* nS;
            id1 = idx_fl + ch_off; % 线性索引
            
            rf_curr = RF_data(:,:,i_ang); 
            
            val = rf_curr(id1) .* (1 - frac) + rf_curr(id1+1) .* frac;
            val = val .* mask .* ApoMask;
            
            slice_accum = slice_accum + sum(val, 2); 
        end
        vol_RC(:, :, iy) = reshape(slice_accum, [scan.N_z, scan.N_x]);
    end
end

% --- [底层核心] 2. CR 重建 (极致向量化，消灭 interp1) ---
function vol_CR = reconstruct_CR_Sliced(RcvData, Trans, Resource, TX, TW, Receive, scan, angle_set, q, baseName)
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
    [nS, nCh, ~] = size(RF_data);
    
    vol_CR = zeros(scan.N_z, scan.N_x, scan.N_y, 'single');
    
    ProbeX = ElementPos(1:128,1).'; 
    ProbeZ = ElementPos(1:128,3).'; 
    D = abs(ProbeX(end)-ProbeX(1));
    offset_distance = TW.peak*lambda;
    
    z_ax = scan.z_axis(:); x_ax = scan.x_axis(:); y_ax = scan.y_axis(:);
    
    for iy = 1:scan.N_y
        % --- 【切片级实时心跳包】 ---
        if mod(iy, 16) == 0 || iy == 1
            send(q, sprintf('      ⏳ [%s] CR 进度: 切片 %d / %d', baseName, iy, scan.N_y));
        end
        
        y_val = y_ax(iy);
        [grid_x, grid_z] = meshgrid(x_ax, z_ax);
        flat_x = grid_x(:); flat_z = grid_z(:);
        
        Px_rel = ProbeX - flat_x; 
        Pz_rel = ProbeZ - flat_z;
        RxDist = sqrt(Px_rel.^2 + Pz_rel.^2);
        
        rx_f_num = 1.5;
        ApoMask = abs(rx_f_num .* Px_rel ./ Pz_rel) <= 0.5;
        
        slice_accum = zeros(length(flat_x), 1, 'single');
        
        for i_ang = 1:length(angle_set)
            n_wave = angle_set(i_ang);
            angle_val = beta(n_wave);
            
            TxDelay = flat_z*cos(angle_val) + y_val*sin(angle_val) + ...
                      (D/2)*sin(angle_val)*sign(angle_val) + offset_distance;
                      
            % --- 纯矩阵运算代替 interp1 ---
            Tidx = (RxDist + TxDelay) .* (fs/c0) + 1;
            
            idx_fl = floor(Tidx);
            frac = Tidx - idx_fl;
            
            mask = (idx_fl >= 1) & (idx_fl < nS);
            idx_fl = max(1, min(nS-1, idx_fl));
            
            ch_off = (0:nCh-1) .* nS;
            id1 = idx_fl + ch_off;
            
            rf_curr = RF_data(:,:,i_ang); 
            
            val = rf_curr(id1) .* (1 - frac) + rf_curr(id1+1) .* frac;
            val = val .* mask .* ApoMask;
            
            slice_accum = slice_accum + sum(val, 2);
        end
        vol_CR(:, :, iy) = reshape(slice_accum, [scan.N_z, scan.N_x]);
    end
end

% --- [局部函数] 3. 数据提取 ---
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

% --- [局部函数] 4. 保存三视图 ---
function save_ortho_views(vol_complex, scan, save_path, title_str)
    try
        vol_abs = abs(vol_complex);
        max_val = max(vol_abs(:)); if max_val==0, max_val=1; end
        vol_db = 20*log10(vol_abs ./ max_val);
        db_min = -60; db_max = 0;
        
        h = figure('Visible', 'off'); 
        set(h, 'Position', [100 100 1200 400]);
        if exist('sgtitle', 'file'), sgtitle(strrep(title_str, '_', '\_'), 'FontSize', 12); end
        
        z_idx = round(scan.N_z / 2); 
        img_xy = squeeze(vol_db(z_idx, :, :));
        subplot(1,3,1);
        imagesc(scan.x_axis*1000, scan.y_axis*1000, img_xy);
        title(sprintf('XY (Z=%.1fmm)', scan.z_axis(z_idx)*1000));
        axis image; colormap gray; caxis([db_min db_max]);
        
        y_idx = round(scan.N_y / 2); 
        img_xz = squeeze(vol_db(:, :, y_idx));
        subplot(1,3,2);
        imagesc(scan.x_axis*1000, scan.z_axis*1000, img_xz);
        title(sprintf('XZ (Y=%.1fmm)', scan.y_axis(y_idx)*1000));
        axis image; colormap gray; caxis([db_min db_max]);
        
        x_idx = round(scan.N_x / 2); 
        img_yz = squeeze(vol_db(:, x_idx, :));
        subplot(1,3,3);
        imagesc(scan.y_axis*1000, scan.z_axis*1000, img_yz);
        title(sprintf('YZ (X=%.1fmm)', scan.x_axis(x_idx)*1000));
        axis image; colormap gray; caxis([db_min db_max]);
        
        saveas(h, save_path);
        close(h);
    catch
    end
end