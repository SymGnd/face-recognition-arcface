%% Reorganize Dataset
sourceDir = 'C:\Users\SS\Downloads\Faces\Faces';
destDir = 'C:\Users\SS\Downloads\Dataset_Organized';

% Create destination if needed
if ~exist(destDir, 'dir')
    mkdir(destDir);
end

% Get all image files
files = dir(fullfile(sourceDir, '*.jpg'));

for i = 1:length(files)
    filename = files(i).name;
    
    % Extract person name (everything before last underscore)
    parts = strsplit(filename, '_');
    personName = strjoin(parts(1:end-1), '_');
    
    % Create person folder if needed
    personDir = fullfile(destDir, personName);
    if ~exist(personDir, 'dir')
        mkdir(personDir);
    end
    
    % Move file
    movefile(fullfile(sourceDir, filename), fullfile(personDir, filename));
end