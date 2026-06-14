% % ==========================================================
% % function_Recon_CR.m (V7 - 修复版 + 动态进度条)
% % ==========================================================
function [volume_CR, x_axis, y_axis, z_axis] = function_Recon_CR(RcvData, Trans, Resource, TX, TW, Receive, angle_set)
% [!!] 注意：第一行必须是 function 定义，不能删掉！

% disp(' [Function] 正在执行: 列发行收 (CR) 重建...'); % (注释掉以免刷屏)

% --- 1. 基本参数 ---
f0 = double(Trans.frequency*1e6);    
fs = f0*Receive(1).samplesPerWave;   
c0 = Resource.Parameters.speedOfSound;
lambda = c0/f0;                      
ElementPos = Trans.ElementPos.*lambda;
initial_time = TW.peak/Trans.frequency/1e6;
pitch = 0.2e-3;   
scan.startdepth = 5e-3;
scan.enddepth = 42e-3;

% --- 2. 提取数据 ---
steer = zeros(length(TX),2);
for n_wave = 1:length(TX)
    steer(n_wave,:) = TX(n_wave).Steer;
end
column_waves = length(TX)/2;
channel_RF = zeros(Receive(1).endSample,Resource.Parameters.numRcvChannels,length(TX),Resource.RcvBuffer.numFrames);
for n_frame = 1:Resource.RcvBuffer.numFrames
    for n_wave = 1:length(TX)
        if iscell(RcvData)
            channel_RF(:,:,n_wave,n_frame) = RcvData{1}(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
        else
            channel_RF(:,:,n_wave,n_frame) = RcvData(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
        end
    end
end
% 提取数据 [1920x33x128]
RF_data = permute(channel_RF(:,1:128,column_waves+1:end,1), [1 3 2]); 
RF_data = hilbert(RF_data);

% --- 3. FK 参数 ---
Nx = 128; dx = 0.0002; Nz = 1024; Nt = Receive(1).endSample;
z_axis_temp = linspace(scan.startdepth,scan.enddepth,Nz).';
dz = z_axis_temp(2)-z_axis_temp(1);
cc = 1540; fHigh = 30e6; fLow = 2;
nFFTt = 8*2^nextpow2(Nt); nFFTx = 2*2^nextpow2(Nx); nFFTz = 2*2^nextpow2(Nz);
dOmega = fs/nFFTt; omega = ifftshift(((0:(nFFTt-1)) - floor(nFFTt/2))'*dOmega);
dkx = 1/dx/nFFTx; kx = ifftshift(((0:(nFFTx-1)) - floor(nFFTx/2))*dkx);
dkz = 1/dz; kz = (fLow)/(cc) + (0:(nFFTz-1))*(dkz/nFFTz); 
omegaBandIndex = (omega <= (fHigh)) & (omega >= (fLow)); 
omegaBand = omega(omegaBandIndex); 
[kx_matrix1,omegaBand_matrix] = meshgrid(kx,omegaBand);
isevanescent = abs(omegaBand_matrix)./abs(kx_matrix1)<cc;

% --- 4. FK 处理 ---
fk_data = zeros(nFFTz,nFFTx,128); 
PkzkxX = zeros(nFFTz,nFFTx,128);

if isnumeric(angle_set)
    angle_indices = angle_set; 
elseif strcmpi(angle_set, 'full')
    angle_indices = 1:column_waves;
elseif strcmpi(angle_set, 'zero')
    angle_indices = 17;
else
    error("未知的 angle_set 参数。");
end

% === [修改开始] 增加动态进度显示 ===
tic;
fprintf(' [CR 处理中]: ');
msg_len = 0; % 用于记录上一条消息的长度
count = 0;   
total_angles = length(angle_indices); 

for ixt = angle_indices
    count = count + 1;
    
    % --- 动态刷新打印 ---
    if msg_len > 0
        fprintf(repmat('\b', 1, msg_len)); % 删除上一行
    end
    msg = sprintf('%d/%d (Idx: %d)', count, total_angles, ixt);
    fprintf(msg);
    msg_len = length(msg); 
    % ---------------------

    % [核心算法]
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
fprintf('\n'); % 换行
fprintf(' [Function] CR 完成, 耗时: %.2f s\n', toc);

% --- 5. 输出 ---
fk_data2 = ifft(ifft(fk_data,[],1),[],2);
volume_CR = fk_data2(140:140+1023, 1:128, :);
z_axis = linspace(5e-3, 42e-3, size(volume_CR,1)); 
x_axis = linspace(-127*pitch/2, 127*pitch/2, 128);    
y_axis = linspace(-127*pitch/2, 127*pitch/2, size(volume_CR,3));
end