# Copyright (C) 2023  Jose Ángel Pérez Garrido
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
import pickle
import os, shutil

def isDir(dirname):
    """
    :param dirname: dirname
    :return: boolean
    """
    return os.path.isdir(dirname)

def make_dirs(path):
    """ Create directory at given path 
    
    :param path: path name
    """
    if not os.path.exists(path):
        os.makedirs(path)

def remove_dirs(path):
    """ Remove directory at given path 
    
    :param path: path name
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def dump_to_json(data, filepath, indent=4, sort_keys=True):
    """ Dump dictionary to json file to a given filepath name 
    
    :param data: python dictionary
    :param filepath: filepath name
    :param indent: indent keys in json file
    :param sort_keys: boolean flag to sort keys
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)

def load_from_json(filepath):
    """ Load json file from filepath into a dictionary 
    
    :param filepath:
    :return: python dictionary
    """
    data = dict()
    with open(filepath) as data_file:    
        data = json.load(data_file)
    return data

def dump_to_txt(data, filepath):
    """ Dump data to txt file format to a given filepath name 
    
    :param filepath: filepath name
    """
    with open(filepath, "w") as file :
        file.write(data)

def dump_to_pickle(data, filepath):
    """ Dump data to pickle format to a given filepath name
    :param filepath: filepath name
    """ 
    with open(filepath, "wb") as file:
        pickle.dump(data, file)