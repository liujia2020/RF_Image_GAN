% =========================================================================
% Convert_NII_Cutoff_Auto.m
% 功能: 读取 NIfTI 文件 -> 执行硬截断 -> 自动重命名 -> 自动存入新文件夹
% 核心: 严格保留原始分辨率 (Spacing/Origin)，仅修改数据值
% =========================================================================

clc; clear; close all;

% =========================================================================
% === [1. 参数配置区 (只改这里)] ===
% =========================================================================

% 输入文件夹 (存放原始 -60dB 的 NII)
InputRoot  = 'E:\Simu_Data\05_Test';

% 目标截断值 (负数)
% 例如: -45 表示将所有小于 -45 的背景噪点全部抹平为 -45
Cutoff_dB  = -30;  

% =========================================================================
% === [2. 自动路径与文件名逻辑] ===
% =========================================================================

% 自动生成输出文件夹名 (例如: E:\Simu_Data\05_Test_Neg45dB)
dirSuffix  = sprintf('_Neg%ddB', abs(Cutoff_dB));
OutputRoot = [InputRoot, dirSuffix];

if ~exist(OutputRoot, 'dir')
    mkdir(OutputRoot);
    fprintf('✅ 自动创建输出文件夹: %s\n', OutputRoot);
else
    fprintf('📂 输出文件夹已存在: %s\n', OutputRoot);
end

fprintf('--- NIfTI 阈值截断工具 启动 ---\n');
fprintf('截断阈值: %d dB\n', Cutoff_dB);

% =========================================================================
% === [3. 批量处理循环] ===
% =========================================================================

files = dir(fullfile(InputRoot, '*.nii'));
nFiles = length(files);

if nFiles == 0
    error('❌ 在输入文件夹里没找到 .nii 文件，请检查路径！');
end

fprintf('发现 %d 个文件，开始处理...\n', nFiles);

for k = 1:nFiles
    rawName = files(k).name;
    srcPath = fullfile(InputRoot, rawName);
    
    % --- 文件名处理 (方案 B: 增加 _45dB 后缀) ---
    [~, nameBody, ext] = fileparts(rawName);
    
    % 如果文件名里已经有 _dB 后缀，先去掉以免重复 (可选)
    % 这里直接追加，确保信息完整
    newName = sprintf('%s_%ddB%s', nameBody, abs(Cutoff_dB), ext);
    dstPath = fullfile(OutputRoot, newName);
    
    fprintf('   [%d/%d] 处理: %s -> %s ... ', k, nFiles, rawName, newName);
    
    try
        % 1. 读取头文件 (关键步骤: 继承原始物理参数)
        info = niftiinfo(srcPath);
        
        % 2. 读取图像数据
        vol = niftiread(info);
        
        % 3. 执行硬截断 (Unified Logic)
        % 将所有小于 Cutoff_dB 的值，统一赋值为 Cutoff_dB
        mask = vol < Cutoff_dB;
        vol(mask) = Cutoff_dB;
        
        % 4. 保存 (带头文件保存，防止分辨率丢失)
        niftiwrite(vol, dstPath, info, 'Compressed', false);
        
        fprintf('✅ 完成\n');
        
    catch ME
        fprintf('❌ 失败: %s\n', ME.message);
    end
end

fprintf('\n🎉 全部转换完毕！\n');
fprintf('请检查输出目录: %s\n', OutputRoot);