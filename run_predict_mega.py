import argparse
import json
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from get_data import get_current_number, spider
from config import *
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="mega", type=str, help="Select training data: Vietlott Mega 6/45")
args = parser.parse_args()

#Turn off eager mode
tf.compat.v1.disable_eager_execution()

def load_model(name):
    """ Load the model"""
    ball_graph = tf.compat.v1.Graph()
    with ball_graph.as_default():
        ball_saver = tf.compat.v1.train.import_meta_graph(
            "{}mega_ball_model.ckpt.meta".format(model_args[args.name]["path"]["ball"])
        )
    ball_sess = tf.compat.v1.Session(graph=ball_graph)
    ball_saver.restore(ball_sess, "{}mega_ball_model.ckpt".format(model_args[args.name]["path"]["ball"]))
    logger.info("Mega ball model loaded!")

    # Load key node names
    with open("{}/{}/{}".format(model_path,args.name , pred_key_name)) as f:
        pred_key_d = json.load(f)

    current_number = get_current_number(args.name)
    logger.info("【{}】Latest issue:{}".format(name_path[args.name]["name"], current_number))
    return ball_graph, ball_sess, pred_key_d, current_number

def build_data_model(name,start,end,mode):
    
    """ Thu thập dữ liệu lịch sử
    :param name Kiểu loại 
    :param start Bắt đầu
    :param end Kết thúc
    :param mode mẫu，train：chế độ luyện tập，predict：chế độ dự đoán（Chế độ đào tạo sẽ giữ các tập tin）
    :return:
    """

    file_path= "{}{}".format(name_path[name]["path"],data_mega_file_name)
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
        })
    return pd.DataFrame(data)
def get_year():
    """ Cut-off year
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])

def try_error(mode, name, predict_features, windows_size):
    """ Handling Exceptions
    """
    if mode:
        return predict_features
    else:
      
        if len(predict_features) != windows_size:
            logger.warning("The issue number has skipped, and the issue number is not continuous! Start looking for the last issue number! The prediction time for this issue is relatively long")
            last_current_year = (get_year() - 1) * 1000
            max_times = 160
            while len(predict_features) != 3:
                predict_features = build_data_model(name, last_current_year + max_times, get_current_number(name), "predict")[[x[0] for x in ball_name]]
                time.sleep(np.random.random(1).tolist()[0])
                max_times -= 1
            return predict_features
        return predict_features
    


def get_mega_ball_predict_result(ball_graph, ball_sess, pred_key_d, predict_features, sequence_len, windows_size):
    """ Nhận kết quả dự đoán bóng đỏ
    """

    name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
    data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
   
    with ball_graph.as_default():
       
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[0][0]])
        print(reverse_sequence)
        pred = ball_sess.run(reverse_sequence, feed_dict={
            "inputs:0": data.reshape(1, windows_size, sequence_len),
            "sequence_length:0": np.array([sequence_len] * 1)
        })
    return pred, name_list



def get_final_result(ball_graph, ball_sess, pred_key_d, name, predict_features, mode=0):
    """" chức năng dự đoán cuối cùng
    """
    m_args = model_args[name]["model_args"]
 
    ball_pred, ball_name_list = get_mega_ball_predict_result(
        ball_graph, ball_sess, pred_key_d,
        predict_features, m_args["ball_sequence_len"], m_args["windows_size"]
    )
   
    ball_name_list = ["{}_{}".format(name[mode], i) for name, i in ball_name_list] + ["{}_{}".format(name[mode], i) for name, i in ball_name_list]
    pred_result_list = ball_pred[0].tolist()
    return {
        b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
    }

def run(name):
    """ Thực hiện dự đoán """
    try:
        ball_graph, ball_sess, pred_key_d, current_number = load_model(name)
        windows_size = model_args[name]["model_args"]["windows_size"]
        data = build_data_model(name, 1, current_number, "predict")
       
        logger.info("【{}】Số kỳ dự báo：{}".format(name_path[name]["name"], datetime.datetime.now()))
        predict_features_ = try_error(1, name, data.iloc[:windows_size], windows_size)
     
        logger.info("Kết quả dự đoán：{}".format(get_final_result(
            ball_graph, ball_sess, pred_key_d, name, predict_features_))
        )
    except Exception as e:
        logger.info("Tải mô hình không thành công, kiểm tra xem mô hình đã được huấn luyện chưa, lỗi: {}".format(e))


if __name__ == '__main__':
    run(args.name)
     