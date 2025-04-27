import os

def organize_folder_into_subfolders(path_to_original_folder, max_number_of_files_per_subfolder=50):
    '''Moves all files in a folder into newly created subfolders comprising of the max_number_of_files_per_subfolder or fewer'''
    files_in_folder = os.listdir(path_to_original_folder)
    if not path_to_original_folder.endswith('/'):
        path_to_original_folder += '/'
    temp_path_to_original_folder = path_to_original_folder + 'temp_folder'
    os.makedirs(temp_path_to_original_folder)
    subfolders_dict = {'temp_subfolder_0': []}
    os.makedirs(temp_path_to_original_folder + '/' + 'temp_subfolder_0')
    for _file_name in files_in_folder:
        if len(subfolders_dict['temp_subfolder_' + str(len(subfolders_dict) - 1)]) == max_number_of_files_per_subfolder:
            subfolders_dict['temp_subfolder_' + str(len(subfolders_dict))] = []
            os.makedirs(temp_path_to_original_folder + '/' + 'temp_subfolder_' + str(len(subfolders_dict) - 1))
        subfolders_dict['temp_subfolder_' + str(len(subfolders_dict) - 1)].append(_file_name)
    for _file_subfolder_path, _file_names in subfolders_dict.items():
        for _file_name in _file_names:
            os.rename(path_to_original_folder + _file_name, temp_path_to_original_folder + '/' + _file_subfolder_path + '/' + _file_name)
    return subfolders_dict

import gdown
url = 'https://drive.google.com/drive/folders/1OXV4qhFF_qJ8VqyrXpR7CzHDsToaqY_W?usp=drive_link'
gdown.download_folder(url, quiet=True, use_cookies=False, remaining_ok=True)