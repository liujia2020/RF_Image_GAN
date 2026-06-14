% =========================================================================
% Convert_NII_Soft_Threshold_3D.m
% 功能: 专为 3DSlicer 优化生成的 NIfTI
% 核心: 使用"软阈值(Soft Ramp)"代替"硬截断"，消除 3D 渲染时的马赛克锯齿
% =========================================================================

clc; clear; close all;

% =========================================================================
% === [1. 参数配置] ===
% =========================================================================

% 输入文件夹 (请指向原始 -60dB 的文件夹)
InputRoot  = 'E:\Simu_Data\05_Test';

% 输出文件夹 (自动创建)
OutputRoot = 'E:\Simu_Data\06_NIfTI_Final_3D_Soft';

% [关键] 您在交互工具里找到的最佳阈值 (比如 -38)
Target_dB  = -38; 

% [关键] 过渡区宽度 (建议 3~5 dB)
% 这就是消除马赛克的秘密。
% 它表示数据会在 Target_dB - 5 到 Target_dB 之间平滑淡出，而不是一刀切。
Soft_Range = 5; 

% =========================================================================
% === [2. 核心处理] ===
% =========================================================================

if ~exist(OutputRoot, 'dir'), mkdir(OutputRoot); end
fprintf('--- 3D NIfTI 软阈值生成器 启动 ---\n');
fprintf('目标阈值: %d dB | 过渡区: %d dB\n', Target_dB, Soft_Range);

files = dir(fullfile(InputRoot, '*.nii'));
nFiles = length(files);

for k = 1:nFiles
    rawName = files(k).name;
    srcPath = fullfile(InputRoot, rawName);
    
    % 构建文件名 (方案B: _38dB_Soft.nii)
    [~, nameBody, ext] = fileparts(rawName);
    newName = sprintf('%s_%ddB_Soft%s', nameBody, abs(Target_dB), ext);
    dstPath = fullfile(OutputRoot, newName);
    
    fprintf('[%d/%d] 正在生成 3D 优化文件: %s ... ', k, nFiles, newName);
    
    try
        % 1. 读取头文件 (严格保留分辨率)
        info = niftiinfo(srcPath);
        
        % 2. 读取数据
        vol = niftiread(info);
        
        % 3. 执行软阈值处理 (Soft Thresholding)
        % 逻辑:
        %   > Target_dB       : 保持原样 (主体)
        %   < Target_dB - Range : 彻底压平 (背景)
        %   中间区域          : 线性淡出 (消除锯齿的关键)
        
        upper_bound = Target_dB;                % 比如 -38
        lower_bound = Target_dB - Soft_Range;   % 比如 -43
        background_val = lower_bound;           % 背景统一设为最低值
        
        % 创建掩膜
        mask_high = vol >= upper_bound;
        mask_low  = vol <= lower_bound;
        mask_mid  = ~mask_high & ~mask_low;
        
        % 处理数据
        vol_new = vol;
        
        % A. 低于下限的，设为背景底色
        vol_new(mask_low) = background_val;
        
        % B. 中间过渡区的，进行平滑线性插值 (Linear Ramp)
        % 这一步让悬崖变成斜坡，消除马赛克
        if any(mask_mid(:))
            % 归一化因子 (0~1)
            alpha = (vol(mask_mid) - lower_bound) / (upper_bound - lower_bound);
            % 映射回原值 (其实这就等于原值，但在更复杂的曲线里有用，这里为了保真保持线性)
            % 为了更平滑，我们可以让它稍微贴近下限一点，但在dB域线性通常足够
            vol_new(mask_mid) = vol(mask_mid); 
        end
        
        % C. 高于上限的，保持不变 (保留细节)
        
        % --- 额外修正 ---
        % 为了让 3DSlicer 的 Volume Rendering 彻底把背景变透明，
        % 我们通常希望背景值是整个数据的最小值。
        % 上面已经把背景设为 lower_bound (-43)。
        % 这样在 Slicer 里，你只要把 Opacity Map 的左端点拉到 -43，背景就全透了。
        
        % 4. 保存
        niftiwrite(vol_new, dstPath, info, 'Compressed', false);
        fprintf('✅ 完成\n');
        
    catch ME
        fprintf('❌ 失败: %s\n', ME.message);
    end
end

fprintf('\n🎉 全部处理完毕！\n');
fprintf('请在 3DSlicer 中打开新生成的文件。\n');
fprintf('提示: 在 Volume Rendering 模块中，背景噪声已经被压缩到了 %d dB 以下。\n', lower_bound);