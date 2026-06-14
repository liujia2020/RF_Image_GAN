% % =========================================================================
% % Run_Selectable_Recon_V2.m 
% % (支持开关 + 进度条 + 物理角度 + 统一保存路径)
% % =========================================================================
clc; clear; close all;

% =========================================================================
% === [1. 路径与配置] ===
% =========================================================================
inputDir = 'D:\Simu_Data_600_1500\02_RF_Data'; 

% [!!] 核心修改：定义统一输出根目录
MasterReconDir = 'D:\Simu_Data_600_1500\03_Recon_Mat';
if ~exist(MasterReconDir, 'dir'), mkdir(MasterReconDir); end

% --- 开关: 控制本次跑哪些 ---
DO_SQ = 1;   % SQ (75角, GT)
DO_HQ = 1;   % HQ (33角, 对照)
DO_LQ = 1;   % LQ (3角,  稀疏输入 - 3D叠加版)

% =========================================================================
% === [2. 索引与角度定义] PICMUS 75角度标准 ===
% ======================s===================================================
na = 75; 
angle_span = 32; % -16 到 +16 度
PhysicalAngles = linspace(-angle_span/2, angle_span/2, na);

% --- 定义索引 ---
idx_SQ = 1:75;
idx_HQ = [1, 3, 6, 8, 10, 13, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, ...
          38, ... 
          40, 43, 45, 47, 50, 52, 54, 57, 59, 61, 63, 66, 68, 70, 73, 75];
idx_LQ = [3, 38, 73]; 

% --- 构建任务 ---
Tasks = struct('name', {}, 'indices', {}, 'folder', {});
cnt = 0;
if DO_SQ, cnt=cnt+1; Tasks(cnt).name='SQ'; Tasks(cnt).indices=idx_SQ; Tasks(cnt).folder='Recon_SQ_75'; end
if DO_HQ, cnt=cnt+1; Tasks(cnt).name='HQ'; Tasks(cnt).indices=idx_HQ; Tasks(cnt).folder='Recon_HQ_33'; end
if DO_LQ, cnt=cnt+1; Tasks(cnt).name='LQ'; Tasks(cnt).indices=idx_LQ; Tasks(cnt).folder='Recon_LQ_03'; end

if isempty(Tasks), error('请至少选择一个任务 (DO_SQ/HQ/LQ = true)'); end

% =========================================================================
% === [3. 开始处理] ===
% =========================================================================
fprintf('\n--- 任务配置 (输出至: %s) ---\n', MasterReconDir);
for t = 1:length(Tasks)
    % 预创建子文件夹
    targetSubDir = fullfile(MasterReconDir, Tasks(t).folder);
    if ~exist(targetSubDir, 'dir'), mkdir(targetSubDir); end
    fprintf(' [%d] %s: %d 个角度 -> %s\\\n', t, Tasks(t).name, length(Tasks(t).indices), Tasks(t).folder);
end

fileList = dir(fullfile(inputDir, '**', '*.mat'));
validMask = ~startsWith({fileList.name}, '.') & ~startsWith({fileList.name}, 'Recon_'); 
fileList = fileList(validMask);
totalFiles = length(fileList);
fprintf('--------------------------------------\n');
fprintf('找到 %d 个 RF 文件。开始...\n', totalFiles);
%%
GlobalTimer = tic;
barLength = 30; 

for i = 1 : totalFiles
    fileInfo = fileList(i);
    [~, baseName, ~] = fileparts(fileInfo.name);
    fullPath = fullfile(fileInfo.folder, fileInfo.name);
    
    % --- 进度条 ---
    percent = i / totalFiles;
    nBars = floor(percent * barLength);
    progBar = ['[', repmat('=', 1, nBars), repmat(' ', 1, barLength - nBars), ']'];
    fprintf('\n%s %3d%% (%d/%d) | 文件: %s\n', progBar, round(percent*100), i, totalFiles, baseName);
    
    try, M = load(fullPath, 'Resource'); nFrames = M.Resource.RcvBuffer(1).numFrames;
    catch, fprintf(['   [跳过] 无法读VSX' ...
            '取 Resource\n']); continue; end
    
    if nFrames==20, t_frms=[1]; elseif nFrames==10, t_frms=[1]; 
    elseif nFrames==4, t_frms=[1]; else, t_frms=1; end
    
    try, D = load(fullPath); catch, fprintf('   [坏文件] 跳过\n'); continue; end
    if iscell(D.RcvData), RcvAll=D.RcvData{1}; else, RcvAll=D.RcvData; end
    
    for frm = t_frms
        RF_Single = RcvAll(:, :, frm);
        Res_S = D.Resource; Res_S.RcvBuffer(1).numFrames = 1;
        
        nEvt = length(D.Receive)/nFrames;
        Rec_S = D.Receive((frm-1)*nEvt+1 : frm*nEvt);
        for m=1:length(Rec_S), Rec_S(m).framenum = 1; end
        
        % --- 遍历任务 ---
        for t = 1:length(Tasks)
            currTask = Tasks(t);
            idx = currTask.indices;
            
            % [!!] 核心修改：使用统一的 MasterReconDir
            outDir = fullfile(MasterReconDir, currTask.folder);
            % (无需每次mkdir，循环开始前已建好)
            
            outFile = fullfile(outDir, sprintf('Recon_%s_Frm_%02d_%s.mat', baseName, frm, lower(currTask.name)));
            if exist(outFile, 'file'), continue; end
            
            % 打印角度信息
            current_angles = PhysicalAngles(idx);
            if length(idx) <= 5
                angleStr = sprintf('%.1f ', current_angles);
                msg = sprintf('   -> [%s] Frm %d | 角度: [%s]', currTask.name, frm, angleStr);
            else
                msg = sprintf('   -> [%s] Frm %d | 角度: %.1f° ... %.1f° (共%d个)', ...
                    currTask.name, frm, current_angles(1), current_angles(end), length(idx));
            end
            fprintf('%s\n', msg);
            
            [vol_CR, x, y, z] = function_Recon_CR(RF_Single, D.Trans, Res_S, D.TX, D.TW, Rec_S, idx);
            [vol_RC, ~, ~, ~] = function_Recon_RC(RF_Single, D.Trans, Res_S, D.TX, D.TW, Rec_S, idx);
            volume_final = vol_RC + permute(vol_CR, [1 3 2]);
            
            save(outFile, 'volume_final', 'x', 'y', 'z');
        end
    end
end

fprintf('\n全部完成！耗时 %.1f min\n', toc(GlobalTimer)/60);