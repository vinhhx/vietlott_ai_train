import argparse
import json
import time
import datetime
import numpy as np
import tensorflow as tf
from get_data import get_current_number, spider
from config import *
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="Select training data: Double Color Ball/Lotto")
args = parser.parse_args()

#Turn off eager mode
tf.compat.v1.disable_eager_execution()

def load_model(name):
    """ Load the model"""
    if name == "ssq":
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(model_args[args.name]["path"]["red"])
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(model_args[args.name]["path"]["red"]))
        logger.info("Red ball model loaded!")

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(model_args[args.name]["path"]["blue"])
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(model_args[args.name]["path"]["blue"]))
        logger.info("The blue ball model has been loaded!")

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path, args.name, pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info("【{}】Latest Issue:{}".format(name_path[args.name]["name"], current_number))
        return red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number
    else:
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(model_args[args.name]["path"]["red"])
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(model_args[args.name]["path"]["red"]))
        logger.info("Red ball model loaded!")

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(model_args[args.name]["path"]["blue"])
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(model_args[args.name]["path"]["blue"]))
        logger.info("The blue ball model has been loaded!")

        # Load key node names
        with open("{}/{}/{}".format(model_path,args.name , pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info("【{}】Latest issue:{}".format(name_path[args.name]["name"], current_number))
        return red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number

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
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict")[[x[0] for x in ball_name]]
                time.sleep(np.random.random(1).tolist()[0])
                max_times -= 1
            return predict_features
        return predict_features
    


def get_red_ball_predict_result(red_graph, red_sess, pred_key_d, predict_features, sequence_len, windows_size):
    """ Nhận kết quả dự đoán bóng đỏ
    """

    name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
    data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
   
    with red_graph.as_default():
       
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[0][0]])
        print(reverse_sequence)
        pred = red_sess.run(reverse_sequence, feed_dict={
            "inputs:0": data.reshape(1, windows_size, sequence_len),
            "sequence_length:0": np.array([sequence_len] * 1)
        })
    return pred, name_list


def get_blue_ball_predict_result(blue_graph, blue_sess, pred_key_d, name, predict_features, sequence_len, windows_size):
    """ Nhận kết quả dự đoán bóng rổ
    """
    if name == "ssq":
        data = predict_features[[ball_name[1][0]]].values.astype(int) - 1
        with blue_graph.as_default():
            softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[1][0]])
            pred = blue_sess.run(softmax, feed_dict={
                "inputs:0": data.reshape(1, windows_size)
            })
        return pred
    else:
        name_list = [(ball_name[1], i + 1) for i in range(sequence_len)]
        data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
        with blue_graph.as_default():
            reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[1][0]])
            pred = blue_sess.run(reverse_sequence, feed_dict={
                "inputs:0": data.reshape(1, windows_size, sequence_len),
                "sequence_length:0": np.array([sequence_len] * 1)
            })
        return pred, name_list

def get_final_result(red_graph, red_sess, blue_graph, blue_sess, pred_key_d, name, predict_features, mode=0):
    """" chức năng dự đoán cuối cùng
    """
    m_args = model_args[name]["model_args"]
    if name == "ssq":
      
        red_pred, red_name_list = get_red_ball_predict_result(
            red_graph, red_sess, pred_key_d,
            predict_features, m_args["sequence_len"], m_args["windows_size"]
        )
        blue_pred = get_blue_ball_predict_result(
            blue_graph, blue_sess, pred_key_d,
            name, predict_features, 0, m_args["windows_size"]
        )
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [ball_name[1][mode]]
        pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }
    else:
        red_pred, red_name_list = get_red_ball_predict_result(
            red_graph, red_sess, pred_key_d,
            predict_features, m_args["red_sequence_len"], m_args["windows_size"]
        )
       
       
        blue_pred, blue_name_list = get_blue_ball_predict_result(
            blue_graph, blue_sess, pred_key_d,
            name, predict_features, m_args["blue_sequence_len"], m_args["windows_size"]
        )
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
        pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }

def run(name):
    """ Thực hiện dự đoán """
    try:
        red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number = load_model(name)
        windows_size = model_args[name]["model_args"]["windows_size"]
        data = spider(name, 1, current_number, "predict")

        logger.info("【{}】Số kỳ dự báo：{}".format(name_path[name]["name"], int(current_number) + 1))
        predict_features_ = try_error(1, name, data.iloc[:windows_size], windows_size)
     
        logger.info("Kết quả dự đoán：{}".format(get_final_result(
            red_graph, red_sess, blue_graph, blue_sess, pred_key_d, name, predict_features_))
        )
    except Exception as e:
        logger.info("Tải mô hình không thành công, kiểm tra xem mô hình đã được huấn luyện chưa, lỗi: {}".format(e))


if __name__ == '__main__':
    if not args.name:
        run_name="dlt"
    else:
        run_name = args.name
    run(run_name)
     