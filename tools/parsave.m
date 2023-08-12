function parsave(savePath, fileName, fileFormat, saveFile)
%% This is actually a function used to save files, 
%% but it can be used in the parfor loop.

save(fullfile(savePath, strcat(fileName, fileFormat)), 'saveFile');

end