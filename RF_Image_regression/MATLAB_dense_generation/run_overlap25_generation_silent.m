cd(fileparts(mfilename('fullpath')));

log_path = fullfile(pwd, 'overlap25_generation_evalc_log.txt');
status_path = fullfile(pwd, 'overlap25_generation_status.txt');

status = 'FAILED';
log_text = '';

try
    log_text = evalc('run(''build_RF_dense_carotid_test_32x16x16.m'')');
    status = 'OK';
catch ME
    log_text = sprintf('%s\n\n%s\n', log_text, getReport(ME, 'extended', 'hyperlinks', 'off'));
end

fid = fopen(log_path, 'w');
if fid < 0
    error('Could not open log file: %s', log_path);
end
fprintf(fid, '%s', log_text);
fclose(fid);

fid = fopen(status_path, 'w');
if fid < 0
    error('Could not open status file: %s', status_path);
end
fprintf(fid, '%s\n', status);
fclose(fid);

if ~strcmp(status, 'OK')
    error('Overlap25 generation failed. See %s', log_path);
end
