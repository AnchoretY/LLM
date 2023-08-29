'''
Author: AnchoretY
Date: 2023-05-23 03:50:42
LastEditors: AnchoretY
LastEditTime: 2023-05-31 07:15:37
'''
'''
Author: Yhk
Date: 2022-05-15 22:53:43
LastEditors: AnchoretY
LastEditTime: 2023-05-23 03:49:56
Description: 
'''
import os
import glob
import json
import pickle
import shutil
import pandas as pd
from tqdm import tqdm
import logging
import hashlib



def count_file_line(path):
    """
        获取函数行数
    """
    res = os.popen(f'wc -l {path}').readlines()
    if res == []: 
        raise Exception('file line not countable')
    try:
        num_line = int(res[0].split()[0])
        return num_line
    except:
        raise Exception('file line not countable')


def read_datapath_file(dataset_path):
    """
        读取数据集路径，获得其中包含的cfg文件列表
    """
    file_l = []
    for program in os.listdir(dataset_path):
        program_path = os.path.join(dataset_path,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                for filename in cfg_filename_l:
                    file = os.path.join(complie_path,filename)
                    file_l.append(file)
    return file_l

def read_json(file):
    with open(file,'r') as load_f:
         load_dict = json.load(load_f)
    return load_dict

def read_pickle(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data,file):
    with open(file,'wb') as f:
        data = pickle.dump(data,f)
    

def get_function_name(file,part_nums=3):
    return file.split("/")[-1].split("_",part_nums-1)[-1]
    
def get_function_origin_info(file):
    func_name = os.path.split(file)[1].split(".")[0]
    path,complie_info = os.path.split(os.path.dirname(file))
    word_size,optimizer,complier = complie_info.split("_")
    path,library = os.path.split(path)
    _,program = os.path.split(path)
    return func_name,program,library,word_size,optimizer,complier

def get_file_list(file_path):

    file_l = []
    for program in os.listdir(file_path):
        program_path = os.path.join(file_path,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                for filename in cfg_filename_l:
                    file = os.path.join(complie_path,filename)
                    file_l.append(file)
    return file_l

def get_group_names(file_id,df_group):
    """
    根据文件id在df_group中查找该文件所在组的id
    """
    group_names = []
    for i,row in df_group.iterrows():
        this_group_names = row.to_list()
        if file_id in this_group_names:
            group_names =  [x for x in this_group_names if pd.isnull(x) == False]
            break
    group_names.remove(file_id)
    return group_names

def get_group_index(file_id,df_group):
    """
    根据文件id在df_group中查找该文件所在组的id
    """
    for group_id,row in df_group.iterrows():
        this_group_names = row.to_list()
        if file_id in this_group_names:
            break
    return group_id

def get_file_group_map(group_file,with_gid=True,fid_type=int):
    """
        Return:
            group_to_file_map:组到组内fid列表的映射
            file_to_group_map：fid到所属组的映射
    """
    logging.info("Read File Group Info...")
    group_to_file_map = {}
    file_to_group_map = {}
    with open(group_file,'r') as f:
        for i,line in tqdm(enumerate(f),total=count_file_line(group_file),desc="Load File Group Map:"):
            line = line.strip().split(",")
            line = list(set(map(lambda x: fid_type(x), line)))

            if with_gid:
                gid,fids = line[0],line[1:]
            else:
                gid,fids = i,line
            
            group_to_file_map[gid]=fids
            for fid in fids:
                file_to_group_map[fid] = gid
            
    logging.info("Read File Group Info Completed!")
    return group_to_file_map,file_to_group_map


# 获得函数fid->info的map
def get_func_info_map(func_file):
    func_info_map ={}
    with open(func_file,"r") as f:
        for line in tqdm(f.readlines(),total=count_file_line(func_file),desc="Load Func Info Map:"):
            data = json.loads(line)
            func_info_map[data['fid']] = data
    return func_info_map

# def get_file_group_map(group_file):
#     """
#         Args:
#             group_file: 分组文件地址
#         Return:
#             group_to_file_map: group_id映射到所属组包含的file_id字典
#             file_to_group_map: file_id映射到所属group_id的字典
#     """
#     logging.info("Read File Group Info...")
#     df_group = pd.read_csv(group_file,names=range(62),header=None)
#     group_to_file_map = {}
#     file_to_group_map = {}
#     for _,row in tqdm(df_group.iterrows(),total=df_group.shape[0],desc="Load File Group Map:"):
#         row_data = row.tolist()
#         group_id,row_data = int(row_data[0]),row_data[1:]
#         file_l = [int(x) for x in row_data if pd.isnull(x) == False]
#         group_to_file_map[group_id] = file_l
#         for file in file_l:
#             file_to_group_map[file]=group_id
#     logging.info("Read File Group Info Completed!")
#     return group_to_file_map,file_to_group_map

def create_path_not_exists(save_file):
    """
        如果文件路径或文件所在路径不存在，则创建目录
        save_file: 存储文件的文件路径
    """
    path = os.path.dirname(save_file)
    if not os.path.exists(path):
        os.makedirs(path)
        print("Create Dir:{}".format(path))
    
def init_path(path):
    """
        初始化一个路径，如果路径存在则情况，路径不存在则创建目录
    """
    if not os.path.exists(path):
        logging.info("Create path:{}".format(path))
        os.makedirs(path)
    else:
        logging.info("Clear Path:{}".format(path))
        shutil.rmtree(path) 
        os.makedirs(path)

def compute_md5_hash(my_string):
    # Create an MD5 hash object
    hash_object = hashlib.md5()

    # Update the hash object to include the string
    hash_object.update(my_string.encode())

    # Get the hexadecimal representation of the hash value
    hash_hex = hash_object.hexdigest()

    return hash_hex