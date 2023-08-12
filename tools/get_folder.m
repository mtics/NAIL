function[folder] = get_folder(folder)
% GET_FOLDER used to get the specified folder.
% If the folder is not existed, it will be created
%
% Input:
%   folder: str, the spceified folder.
%
% Output:
%   folder: str, the spceified folder.
%
%
%
% Call:
%   [folder] = get_folder(folder)

    if ~exist(folder, 'dir')
        mkdir(folder);
    end
end