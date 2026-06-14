% =========================================================================
% Show_Comparison_v2.m
% 功能: 绘制对比图 (强制 3D 模式)
%       Row 1: 原始范围 (-60dB)
%       Row 2: 模拟截断 (-30dB)
% =========================================================================

clc; clear; close all;

% === [配置区] ===
filePath = 'G:\DAS\Recon_SimData_Stand_Frm_01_sq.mat';
db_val_1 = -60;  % 原始显示下限
db_val_2 = -40;  % 截断显示下限

% === [读取数据] ===
if ~exist(filePath, 'file'), error('❌ 找不到文件: %s', filePath); end
fprintf('正在读取: %s ...\n', filePath);
D = load(filePath);

% 自动寻找图像变量
raw_vol = [];
if isfield(D, 'volume_final'), raw_vol = D.volume_final;
elseif isfield(D, 'volume_combined'), raw_vol = D.volume_combined;
else
    vars = fieldnames(D); maxSize = 0;
    for k = 1:length(vars)
        tmp = D.(vars{k});
        if isnumeric(tmp) && ndims(tmp) >= 3 && numel(tmp) > maxSize
            raw_vol = tmp; maxSize = numel(tmp);
        end
    end
end
if isempty(raw_vol), error('❌ 没找到 3D 图像数据'); end

% --- 坐标轴与单位处理 ---
if isfield(D, 'x'), x = D.x; else, x = 1:size(raw_vol,2); end
if isfield(D, 'y'), y = D.y; else, y = 1:size(raw_vol,3); end
if isfield(D, 'z'), z = D.z; else, z = 1:size(raw_vol,1); end

% 智能单位判断: 如果数值很小(<0.01)，说明是米，乘以1000转毫米
if mean(abs(diff(x))) < 0.01, x = x * 1000; end 
if mean(abs(diff(y))) < 0.01, y = y * 1000; end
if mean(abs(diff(z))) < 0.01, z = z * 1000; end

dims = size(raw_vol);
fprintf('数据维度: [%s]\n', num2str(dims));

% --- 信号处理 (包络 + 对数) ---
if ~isreal(raw_vol)
    fprintf('数据为复数，直接取模...\n');
    vol_env = abs(raw_vol);
else
    fprintf('数据为实数 RF，执行 Hilbert 变换...\n');
    vol_env = abs(hilbert(raw_vol));
end

max_val = max(vol_env(:)); 
if max_val == 0, max_val = 1; end
vol_db_raw = 20 * log10(vol_env / max_val + 1e-9);

% --- 准备对比数据 ---
vol_cut = vol_db_raw;
vol_cut(vol_cut < db_val_2) = db_val_2; % 硬截断

% =========================================================================
% === [绘图逻辑 - 强制 3D 三视图] ===
% =========================================================================
figure('Name', 'DB对比分析 (3D版)', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
colormap(gray);

fprintf('正在绘制 3D 三视图对比...\n');

% 计算切片索引
% idx_z = max(1, round(dims(1) / 2)); % 深度中间
idx_z = 500;
idx_x = max(1, round(dims(2) / 2)); % 宽度中间
idx_y = max(1, round(dims(3) / 2)); % 高度中间

% === 第一行: 原始数据 (-60dB) ===
plot_row(1, vol_db_raw, x, y, z, idx_x, idx_y, idx_z, db_val_1, '原始');

% === 第二行: 截断数据 (-30dB) ===
plot_row(2, vol_cut,    x, y, z, idx_x, idx_y, idx_z, db_val_2, '硬截断');

fprintf('✅ 完成！\n');

% =========================================================================
% 绘图辅助函数 (直接定义在脚本末尾)
% =========================================================================
function plot_row(row_idx, vol, x, y, z, ix, iy, iz, db_range, tag)
    base = (row_idx - 1) * 3;
    
    % 1. 正视图 XZ (B-mode 视角)
    subplot(2, 3, base + 1);
    slice_xz = squeeze(vol(:, :, iy));
    imagesc(x, z, slice_xz);
    axis image; caxis([db_range, 0]); colorbar;
    title([tag ' XZ (正视图)'], 'FontSize', 12);
    xlabel('Width (mm)'); ylabel('Depth (mm)');
    
    % 2. 侧视图 YZ
    subplot(2, 3, base + 2);
    slice_yz = squeeze(vol(:, ix, :));
    imagesc(y, z, slice_yz);
    axis image; caxis([db_range, 0]); colorbar;
    title([tag ' YZ (侧视图)'], 'FontSize', 12);
    xlabel('Elevation (mm)'); ylabel('Depth (mm)');
    
    % 3. 俯视图 XY (C-scan 视角)
    subplot(2, 3, base + 3);
    slice_xy = squeeze(vol(iz, :, :)).'; % 转置让Y轴竖直
    imagesc(x, y, slice_xy);
    axis image; caxis([db_range, 0]); colorbar;
    title([tag ' XY (俯视图)'], 'FontSize', 12);
    xlabel('Width (mm)'); ylabel('Elevation (mm)');
end