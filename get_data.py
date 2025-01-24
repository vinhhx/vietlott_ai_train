import argparse
import requests
import pandas as pd
from loguru import logger
from config import os,name_path,data_file_name

parser = argparse.ArgumentParser()
parser.add_argument('--name',default='ssq',type=str,help='')
args= parser.parse_args()



def get_current_number(name):
    """ Nhận số phát hành mới nhất
    :return: int
    """
    return 10000

def spider(name,start,end,mode):
    """ Thu thập dữ liệu lịch sử
    :param name Kiểu loại 
    :param start Bắt đầu
    :param end Kết thúc
    :param mode mẫu，train：chế độ luyện tập，predict：chế độ dự đoán（Chế độ đào tạo sẽ giữ các tập tin）
    :return:
    """

    file_path= "{}{}".format(name_path[name]["path"],data_file_name)
    data_in_file = pd.read_csv(file_path,header=0)
    data=[]
    for row in data_in_file.values:
        data.append({
            'Day':row[1],
            'Num_1':row[2],
            'Num_2':row[3],
            'Num_3':row[4],
            'Num_4':row[5],
            'Num_5':row[6],
            'Num_6':row[7],
            'NumS_1':row[8],
        })
    return pd.DataFrame(data)

def run(name):
    '''
        :param name: Tên loại sổ xố
        :return:
    '''
    current_number = get_current_number(name)
    logger.info("【{}】Số phát hành mới nhất：{}".format(name_path[name]["name"], current_number))
    logger.info("Nhận【{}】dữ liệu。。。".format(name_path[name]["name"]))
    if not os.path.exists(name_path[name]["path"]):
        os.makedirs(name_path[name]["path"])
    data = spider(name,1,current_number,"train")
    
    if "data" in os.listdir(os.getcwd()):
        logger.info("【{}】Dữ liệu đã sẵn sàng，Tổng cộng{}giai đoạn này, Mô hình có thể được huấn luyện ở bước tiếp theo...".format(name_path[name]["name"], len(data)))
    else:
        logger.error("Tệp dữ liệu không tồn tại")
if __name__ == '__main__':
    if not args.name:
        nameArgs='ssq'
    else:
        nameArgs=args.name

    run(name=nameArgs)

