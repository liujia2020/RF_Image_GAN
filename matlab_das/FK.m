clc; clear; close all;

% =========================================================================
% === [0. 突破 MATLAB 限制，强制指定多核] ===
% =========================================================================
targetWorkers = 50; % 建议用 50 核跑，留些资源给系统

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
OutputRoot = 'G:\Data_0110_RFdata\Carotid_Data\03_FK_Result\';  % 建议改个名字区分 DAS 结果

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
taskIdx = 1; 

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
    taskIdx = taskIdx + 1; 
end

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
% === [2. 解决硬盘风暴：全内存预加载 (RAM Cache)] ===
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
    raw = load(fullPath); 
    
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

q = parallel.pool.DataQueue;
afterEach(q, @(msg) fprintf('%s\n', msg));

%% =========================================================================
% === [3. 阶段 2/2: 并行循环 (0 IO 延迟计算)] ===
% =========================================================================
parfor i = 1 : N_Files
    
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
            continue; 
        end
        
        % 1. 重建 RC (FK 频域法)
        vol_RC = fk_recon_RC(RF_Single, Trans, Res, TX, TW, Rec_S, currTask.indices, q, baseName);
        
        % 2. 重建 CR (FK 频域法)
        vol_CR = fk_recon_CR(RF_Single, Trans, Res, TX, TW, Rec_S, currTask.indices, q, baseName);
        
        % 3. 融合
        volume_final = vol_RC + vol_CR; 
        
        % 4. 保存 MAT
        par_save_mat(MatPath, volume_final, scan.x_axis, scan.y_axis, scan.z_axis);

        % 5. 保存 NIfTI
        vol_abs = abs(volume_final);
        max_val = max(vol_abs(:)); if max_val==0, max_val=1; end
        
        vol_db = 20 * log10(vol_abs ./ max_val + 1e-12);
        vol_db(vol_db < -60) = -60; vol_db(vol_db > 0) = 0;
        vol_nii = single(vol_db); 
        
        dx = abs(scan.x_axis(2)-scan.x_axis(1))*1000; 
        dy = abs(scan.y_axis(2)-scan.y_axis(1))*1000; 
        dz = abs(scan.z_axis(2)-scan.z_axis(1))*1000; 
        vox_size = [dz, dx, dy]; 
        
        if exist(NiiPath, 'file')
            delete(NiiPath); 
        end
        niftiwrite(vol_nii, NiiPath, 'Compressed', false);
        info = niftiinfo(NiiPath);
        info.PixelDimensions = vox_size;
        info.SpaceUnits = 'Millimeter';
        delete(NiiPath); 
        niftiwrite(vol_nii, NiiPath, info, 'Compressed', false);
        
        % 6. 保存三视图
        save_ortho_views(volume_final, scan, PngPath, FileTag);
    end
    
    send(q, sprintf(' ✅✅ [大功告成] 文件进度: %d / %d (%s)', i, N_Files, baseName));
end
fprintf('\n🎉 FK 并行处理全部完成！\n');

% =========================================================================
% === 局部函数区 (必须在最末尾) ===
% =========================================================================

function par_save_mat(fname, volume_final, x, y, z)
    save(fname, 'volume_final', 'x', 'y', 'z', '-v7.3');
end

% --- [核心] FK RC 重建算法 (已适配内存预加载与并行心跳) ---
function vol_RC = fk_recon_RC(RF_Single, Trans, Resource, TX, TW, Receive, angle_set, q, baseName)
    f0 = double(Trans.frequency*1e6);    
    fs = f0*Receive(1).samplesPerWave;   
    c0 = Resource.Parameters.speedOfSound;
    lambda = c0/f0; ElementPos = Trans.ElementPos.*lambda;
    initial_time = TW.peak/Trans.frequency/1e6;
    pitch = 0.2e-3; scan.startdepth = 5e-3; scan.enddepth = 42e-3;

    row_waves = length(TX)/2;
    Nt = Receive(1).endSample;
    
    % 完美衔接内存缓存数据
    RF_data = zeros(Nt, row_waves, 128); 
    for n_wave = 1:row_waves
        startS = Receive(n_wave).startSample;
        endS = Receive(n_wave).endSample;
        RF_data(:, n_wave, :) = RF_Single(startS:endS, Trans.Connector(129:256));
    end
    RF_data = hilbert(RF_data);

    Nx = 128; dx = 0.0002; Nz = 1024;
    z_axis_temp = linspace(scan.startdepth,scan.enddepth,Nz).'; dz = z_axis_temp(2)-z_axis_temp(1);
    cc = 1540; fHigh = 30e6; fLow = 2;
    nFFTt = 8*2^nextpow2(Nt); nFFTx = 2*2^nextpow2(Nx); nFFTz = 2*2^nextpow2(Nz);
    dOmega = fs/nFFTt; omega = ifftshift(((0:(nFFTt-1)) - floor(nFFTt/2))'*dOmega);
    dkx = 1/dx/nFFTx; kx = ifftshift(((0:(nFFTx-1)) - floor(nFFTx/2))*dkx);
    dkz = 1/dz; kz = (fLow)/(cc) + (0:(nFFTz-1))*(dkz/nFFTz); 
    omegaBandIndex = (omega <= (fHigh)) & (omega >= (fLow)); 
    omegaBand = omega(omegaBandIndex); 
    [kx_matrix1,omegaBand_matrix] = meshgrid(kx,omegaBand);
    isevanescent = abs(omegaBand_matrix)./abs(kx_matrix1)<cc;

    fk_data = zeros(nFFTz,nFFTx,128); PkzkxX = zeros(nFFTz,nFFTx,128);
    total_angles = length(angle_set);
    count = 0;

    for ixt = angle_set
        count = count + 1;
        % 发送并行心跳包
        send(q, sprintf('      ⏳ [%s] RC (FK) 进度: 角度 %d / %d (Idx: %d)', baseName, count, total_angles, ixt));

        RFshow = RF_data(1:Nt,ixt,1:128);
        RFo_y = fft(RFshow,nFFTt,1); RFo_y = RFo_y(omegaBandIndex,:); RFo_ky = fft(RFo_y,nFFTx,2);
        
        angle = TX(ixt).Steer(1);
        sinA = sin(angle); cosA = cos(angle);
        dt = -initial_time*cosA+sinA*((Nx-1)*(angle<0)-(0:Nx-1))*dx/cc;
        tmp = bsxfun(@times, omegaBand, dt); 
        [kx_matrix,kz_matrix] = meshgrid(kx,kz);
        if sinA==0, K_all = (kx_matrix.^2+kz_matrix.^2)./(2.*kx_matrix.*0+2.*kz_matrix);
        else, K_all = (-kz_matrix*cosA+sqrt(kx_matrix.^2*sinA.^2+kz_matrix.^2))/sinA.^2; end
        OMEGA_st = cc*K_all;
        
        for i = 1:128 
            RFo_ky1 = RFo_ky.*repmat(exp(-2*1i*pi*tmp(:,i)),[1,nFFTx]);
            RFo_ky1(isevanescent) = 0;
            Pkzkx = zeros(size(OMEGA_st));
            for j=1:nFFTx, Pkzkx(:,j) = interp1(omegaBand,RFo_ky1(:,j),OMEGA_st(:,j)); end
            Akzkx = kz_matrix./K_all; Akzkx(isnan(Akzkx)) = 1; Pkzkx = Pkzkx.*Akzkx;
            Pkzkx(isnan(Pkzkx) | (OMEGA_st < omegaBand(1)) | (OMEGA_st > omegaBand(end))) = 0;

            if angle < 0 
                z_intersect = (ElementPos(i,1) - ElementPos(128,1))/tan(angle);
                valid_mask = linspace(scan.startdepth, scan.enddepth, nFFTz) < z_intersect;  
            elseif angle > 0 
                z_intersect = (ElementPos(i,1) - ElementPos(1,1))/tan(angle);
                valid_mask = double(linspace(scan.startdepth, scan.enddepth, nFFTz)) < z_intersect;  
            else 
                valid_mask = ones(1, nFFTz);
            end
            Pkzkx_spatial = ifft(ifft(Pkzkx,[],1),[],2);  
            Pkzkx_spatial = Pkzkx_spatial.*repmat(valid_mask', 1, size(Pkzkx_spatial,2));
            PkzkxX(:,:,i) = fft(fft(Pkzkx_spatial,[],1),[],2);
        end
        fk_data = fk_data + PkzkxX;
    end
    fk_data2 = ifft(ifft(fk_data,[],1),[],2);
    vol_RC = fk_data2(140:140+1023, 1:128, :);
end

% --- [核心] FK CR 重建算法 (已适配内存预加载与并行心跳) ---
function vol_CR = fk_recon_CR(RF_Single, Trans, Resource, TX, TW, Receive, angle_set, q, baseName)
    f0 = double(Trans.frequency*1e6);    
    fs = f0*Receive(1).samplesPerWave;   
    c0 = Resource.Parameters.speedOfSound;
    lambda = c0/f0; ElementPos = Trans.ElementPos.*lambda;
    initial_time = TW.peak/Trans.frequency/1e6;
    pitch = 0.2e-3; scan.startdepth = 5e-3; scan.enddepth = 42e-3;

    column_waves = length(TX)/2;
    Nt = Receive(1).endSample;
    
    % 完美衔接内存缓存数据
    RF_data = zeros(Nt, column_waves, 128);
    for n_wave = 1:column_waves
        idx_wave = column_waves + n_wave;
        startS = Receive(idx_wave).startSample;
        endS = Receive(idx_wave).endSample;
        RF_data(:, n_wave, :) = RF_Single(startS:endS, Trans.Connector(1:128));
    end
    RF_data = hilbert(RF_data);

    Nx = 128; dx = 0.0002; Nz = 1024;
    z_axis_temp = linspace(scan.startdepth,scan.enddepth,Nz).'; dz = z_axis_temp(2)-z_axis_temp(1);
    cc = 1540; fHigh = 30e6; fLow = 2;
    nFFTt = 8*2^nextpow2(Nt); nFFTx = 2*2^nextpow2(Nx); nFFTz = 2*2^nextpow2(Nz);
    dOmega = fs/nFFTt; omega = ifftshift(((0:(nFFTt-1)) - floor(nFFTt/2))'*dOmega);
    dkx = 1/dx/nFFTx; kx = ifftshift(((0:(nFFTx-1)) - floor(nFFTx/2))*dkx);
    dkz = 1/dz; kz = (fLow)/(cc) + (0:(nFFTz-1))*(dkz/nFFTz); 
    omegaBandIndex = (omega <= (fHigh)) & (omega >= (fLow)); 
    omegaBand = omega(omegaBandIndex); 
    [kx_matrix1,omegaBand_matrix] = meshgrid(kx,omegaBand);
    isevanescent = abs(omegaBand_matrix)./abs(kx_matrix1)<cc;

    fk_data = zeros(nFFTz,nFFTx,128); 
    PkzkxX = zeros(nFFTz,nFFTx,128);
    
    total_angles = length(angle_set); 
    count = 0;

    for ixt = angle_set
        count = count + 1;
        % 发送并行心跳包
        send(q, sprintf('      ⏳ [%s] CR (FK) 进度: 角度 %d / %d (Idx: %d)', baseName, count, total_angles, ixt));

        RFshow = RF_data(1:Nt,ixt,1:128); 
        RFo_y = fft(RFshow,nFFTt,1);     
        RFo_y = RFo_y(omegaBandIndex,:); 
        RFo_ky = fft(RFo_y,nFFTx,2);     
        
        angle = TX(ixt+column_waves).Steer(2);
        sinA = sin(angle); cosA = cos(angle);
        dt = -initial_time*cosA+sinA*((Nx-1)*(angle<0)-(0:Nx-1))*dx/cc;
        tmp = bsxfun(@times, omegaBand, dt);
        [kx_matrix,kz_matrix] = meshgrid(kx,kz);
        if sinA==0, K_all = (kx_matrix.^2+kz_matrix.^2)./(2.*kx_matrix.*0+2.*kz_matrix);
        else, K_all = (-kz_matrix*cosA+sqrt(kx_matrix.^2*sinA.^2+kz_matrix.^2))/sinA.^2; end
        OMEGA_st = cc*K_all;
        
        for i = 1:128 
            RFo_ky1 = RFo_ky.*repmat(exp(-2*1i*pi*tmp(:,i)),[1,nFFTx]);
            RFo_ky1(isevanescent) = 0;
            Pkzkx = zeros(size(OMEGA_st));
            for j=1:nFFTx, Pkzkx(:,j) = interp1(omegaBand,RFo_ky1(:,j),OMEGA_st(:,j)); end
            Akzkx = kz_matrix./K_all; Akzkx(isnan(Akzkx)) = 1; Pkzkx = Pkzkx.*Akzkx;
            Pkzkx(isnan(Pkzkx) | (OMEGA_st < omegaBand(1)) | (OMEGA_st > omegaBand(end))) = 0;
            
            if angle < 0 
                z_intersect = (ElementPos(i,1) - ElementPos(128,1))/tan(angle);
                valid_mask = linspace(scan.startdepth, scan.enddepth, nFFTz) < z_intersect;  
            elseif angle > 0 
                z_intersect = (ElementPos(i,1) - ElementPos(1,1))/tan(angle);
                valid_mask = double(linspace(scan.startdepth, scan.enddepth, nFFTz)) < z_intersect;  
            else 
                valid_mask = ones(1, nFFTz);
            end
            Pkzkx_spatial = ifft(ifft(Pkzkx,[],1),[],2);  
            Pkzkx_spatial = Pkzkx_spatial.*repmat(valid_mask', 1, size(Pkzkx_spatial,2));
            PkzkxX(:,:,i) = fft(fft(Pkzkx_spatial,[],1),[],2);
        end
        fk_data = fk_data + PkzkxX; 
    end
    fk_data2 = ifft(ifft(fk_data,[],1),[],2);
    vol_CR = fk_data2(140:140+1023, 1:128, :);
end

% --- [辅助函数] ---
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