%% vantage仿真模式下生成的点目标成像 
% 此程序完成行发射列接受。之前写的这个行和列可能有点问题了
% 日期：2024.6.24
% 名称：RC6gV_SR_33_angles_RC SR：simulation resolution  
% RC:row transmit column receive
%% intital 
clear ;
close all;
clc;
%% 加载仿真的数据
% data_file = 'E:\3D_imaging\RCA\data\raw_data\';
% load([data_file,'RC6gV_SR_33_angles_1_1.mat'],"RcvData","Receive","Resource","TW","TX","Trans")
% load(['RC6gV_SR_33_angles_5.mat']);
load('RC6gV_ER_33_1_angles.mat')
%%
% 常数设置
f0 = double(Trans.frequency*1e6);
fs = f0*Receive(1).samplesPerWave;
c0 = 1540;
lambda = c0/f0;
ElementPos = Trans.ElementPos.*lambda;
%% 行发射和列发射的偏转角度
steer = zeros(length(TX),2);
channel_RF = zeros(Receive(1).endSample,Resource.Parameters.numRcvChannels,length(TX),Resource.RcvBuffer.numFrames);
for n_wave = 1:length(TX)
    steer(n_wave,:) = TX(n_wave).Steer;
end
row_waves = length(TX)/2;
alpha = steer(1:row_waves,1);
%% 处理数据
for n_frame = 1:Resource.RcvBuffer.numFrames
    for n_wave = 1:length(TX)
        channel_RF(:,:,n_wave,n_frame) = RcvData{1}(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
    end
end
RF_data = channel_RF(:,129:256,1:row_waves,1); % 一帧行发射列接受的数据
RF_data= hilbert(RF_data);
%% 定义扫描范围
pitch = 0.2e-3;   % 对于行和列都是一样
scan.N_x = ceil(127*pitch*2/lambda);
scan.N_y = ceil(127*pitch*2/lambda);
scan.x_axis=linspace(-127*pitch/2,127*pitch/2,scan.N_x);
scan.y_axis = linspace(-127*pitch/2,127*pitch/2,scan.N_y);
scan.startdepth = 5e-3;
scan.enddepth = 40e-3;
scan.N_z = ceil((scan.enddepth-scan.startdepth)*2/lambda);
scan.z_axis = linspace(scan.startdepth,scan.enddepth,scan.N_z);
[scan.x,scan.z,scan.y] = meshgrid(scan.x_axis,scan.z_axis,scan.y_axis);
scan.N_pixels = size(scan.x(:),1);
%% 列接受的探头坐标
CRh_probe.x = ElementPos(129:256,1);
CRh_probe.y = ElementPos(129:256,2);
CRh_probe.z = ElementPos(129:256,3);
% 计算列接受时延
Cxm = CRh_probe.x.'-scan.x(:);
Cym = CRh_probe.y.'-scan.y(:);
Czm = CRh_probe.z.'-scan.z(:);
receive_delay=single(sqrt(Cym.^2+Czm.^2)/c0); % 求解列接受延时需要注意把空间点投影到xy面。
% 计算接受孔径 
rx_f_number = 1.5;
% 列接受变迹
Creceive_apodization= single(abs(rx_f_number.*Cym./Czm)<=0.5); % 此处计算不需要计算x到列接受的距离。
%% 数据延时
D = abs(CRh_probe.y(end)-CRh_probe.y(1));
offset_distance=TW.peak*lambda;
row_Th_column_Rh_data = zeros(scan.N_pixels,128,1);
initial_time = 0;
time_vector = initial_time+(0:(size(RF_data,1)-1))/fs;
for n_wave = 1:row_waves
    transmit_delay = scan.z(:)*cos(alpha(n_wave))+scan.x(:)*sin(alpha(n_wave))+(D/2)*sin(alpha(n_wave))*sign(alpha(n_wave))+offset_distance; 
    for n_rx = 1:128
        delay = receive_delay(:,n_rx)+transmit_delay./c0;
        temp = Creceive_apodization(:,n_rx).*interp1(time_vector,RF_data(:,n_rx,n_wave),delay,'linear',0);
        row_Th_column_Rh_data(:,n_rx,1) =  row_Th_column_Rh_data(:,n_rx,1) +temp;
    end
end
% 生成图像
figure_file = 'E:\3D_imaging\RCA\figure\';
% 绘制z=10e-3,xy面的图
[~,z_index] = min(abs(10e-3-scan.z_axis));
rc_data = sum(row_Th_column_Rh_data,2);              % 行发射列接受的数据通道数据相加           
rc_image = reshape(rc_data,[scan.N_z,scan.N_x,scan.N_y]);  
rc_image_xy = squeeze(rc_image(z_index,:,:));
rc_envelope=abs(rc_image_xy);
rc_envelope=20*log10(rc_envelope./max(rc_envelope(:)));
f1=figure(1);
imagesc(scan.x_axis*1000,scan.y_axis*1000,rc_envelope);
colormap gray;
caxis([-30 0]);axis image;title('DAS');xlabel('Lateral Distance[mm]');ylabel('Elevation Distance[mm]');
set(gca,'FontSize',11);
saveas(f1,[figure_file,'RC6gV_SR_RC_33_1_xy'],'png')
% 绘制y=0,xz面的图
[~,y_index] = min(abs(0e-3-scan.y_axis));
rc_image_xz = squeeze(rc_image(:,:,y_index));
rc_envelope=abs(rc_image_xz);
rc_envelope=20*log10(rc_envelope./max(rc_envelope(:)));
f2=figure(2);
imagesc(scan.x_axis*1000,scan.z_axis*1000,rc_envelope);
colormap gray;
caxis([-60 0]);axis image;title('DAS');xlabel('Lateral Distance[mm]');ylabel('Axial Distance[mm]');
set(gca,'FontSize',11);
saveas(f2,[figure_file,'RC6gV_SR_RC_33_1_xz'],'png')
% 绘制x=0,yz面的图
[~,x_index] = min(abs(0e-3-scan.x_axis));
rc_image_yz = squeeze(rc_image(:,x_index,:));
rc_envelope=abs(rc_image_yz);
rc_envelope=20*log10(rc_envelope./max(rc_envelope(:)));
f3=figure(3);
imagesc(scan.y_axis*1000,scan.z_axis*1000,rc_envelope);
colormap gray;
caxis([-60 0]);axis image;title('DAS');xlabel('Elevation Distance[mm]');ylabel('Axial Distance[mm]');
set(gca,'FontSize',11);
saveas(f3,[figure_file,'RC6gV_SR_RC_33_1_yz'],'png')
%% 保存需要的数据
das_file = 'E:\3D_imaging\RCA\data\das_data\';
save([das_file,'RC6gV_SR_33_1_angles_RC.mat'],'row_Th_column_Rh_data','scan','Creceive_apodization')

%%
% rc_data = sum(row_Th_column_Rh_data, 2);    % [10702692×1×1]
% rc_image = reshape(rc_data, [scan.N_z,scan.N_x,scan.N_y]);    % [273×198×198]
% 
% function img_envelope = process_image(img)
%     img_envelope = abs(img);
%     img_envelope = 20*log10(img_envelope./max(img_envelope(:)));
%     img_envelope(img_envelope < -60) = -60;
% end
% 
% figure(10); % 指定figure编号为10，避免被后续代码覆盖
% set(gcf, 'Position', [100 100 1200 400]);
% sgtitle('行发列收成像结果', 'FontSize', 12);
% 
% % 1. XY平面
% subplot(1,3,1);
% [~,z_index] = min(abs(10e-3-scan.z_axis));
% img_xy = process_image(squeeze(rc_image(z_index,:,:)));
% imagesc(scan.x_axis*1000, scan.y_axis*1000, img_xy);
% title('行发列收 - XY平面');
% xlabel('横向距离[mm]'); ylabel('高程距离[mm]');
% colormap gray; colorbar; axis image;
% caxis([-60 0]);
% 
% % 2. XZ平面
% subplot(1,3,2);
% [~,y_index] = min(abs(0e-3-scan.y_axis));
% img_xz = process_image(squeeze(rc_image(:,:,y_index)));
% imagesc(scan.x_axis*1000, scan.z_axis*1000, img_xz);
% title('行发列收 - XZ平面');
% xlabel('横向距离[mm]'); ylabel('轴向距离[mm]');
% colormap gray; colorbar;
% xlim([-12.7 12.7]); ylim([5 40]);  % 直接设置显示范围
% 
% % 3. YZ平面
% subplot(1,3,3);
% [~,x_index] = min(abs(0e-3-scan.x_axis));
% img_yz = process_image(squeeze(rc_image(:,x_index,:)));
% imagesc(scan.y_axis*1000, scan.z_axis*1000, img_yz);
% title('行发列收 - YZ平面');
% xlabel('高程距离[mm]'); ylabel('轴向距离[mm]');
% colormap gray; colorbar;
% xlim([-12.7 12.7]); ylim([5 40]);  % 直接设置显示范围
% 
% % 统一colorbar范围
% set(findall(gcf,'type','colorbar'),'Limits',[-60 0]);
