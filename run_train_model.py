import time
import json
import argparse
import pandas as pd
import numpy as np
from config import *
from modeling import LstmWithCRFModel,SignalLstmModel,tf
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name',default="ssq",type=str,help="Select training data: Double Color Ball/Lotto")
parser.add_argument('--train_test_split',default=0.7,type=float,help="The proportion of training set should be set greater than 0.5")
args= parser.parse_args()

pred_key ={}

def create_data(data, name, windows):
    """ Creating training data
    :param data: Dataset
    :param name: How to play, Double Color Ball/Lotto
    :param windows: Training Window
    :return:
    """

    if not len(data):
        raise logger.error("Data empty!")
    else:
        #Create Model Folder
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        logger.info("Training data loaded!")
    
    data = data.iloc[:, 2:].values
    logger.info("Training set data dimensions: {}".format(data.shape))
    x_data, y_data = [], []
    for i in range(len(data) - windows -1):
        sub_data = data[i:(i+windows+1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])

    cut_num =6 if name == "ssq" else 6
    return {
        "red": {
            "x_data": np.array(x_data)[:, :, :cut_num], "y_data": np.array(y_data)[:, :cut_num]
        },
        "blue": {
            "x_data": np.array(x_data)[:, :, cut_num:], "y_data": np.array(y_data)[:, cut_num:]
        }
    }

def create_train_test_data(name,windows,train_test_split):
    """ Divide the dataset """
    if train_test_split <0.5:
        raise ValueError("The sampling ratio of the training set is less than 50%, the training is terminated, and resampling is requested （train_test_split>0.5）!")
    path="{}{}".format(name_path[name]["path"],data_file_name)
    data = pd.read_csv(path)
    logger.info("read data from path: {}".format(path))
    
    train_data =create_data(data.iloc[:int(len(data))],name,windows)
    test_data = create_data(data.iloc[int(len(data) * train_test_split):], name, windows)
    # test_data = create_data(data.iloc[lambda x: x.index % 3 == 0],name,windows)
    logger.info(
        "train_data sample total = {}, test_data sample rate = {}".format(int(len(data)),(1 - train_test_split))
    )
    return train_data, test_data


def train_with_eval_red_ball_model(
    name,
    x_train,
    y_train,
    x_test,
    y_test,
):
    """Red ball model training and evaluation"""
    m_args = model_args[name]
    x_train = x_train - 1
    y_train = y_train - 1
    train_data_len = x_train.shape[0]
    
    logger.info("Training feature data dimensions: {}".format(x_train.shape))
    logger.info("Training label data dimensions: {}".format(y_train.shape))

    x_test = x_test - 1
    y_test = y_test - 1 

    test_data_len = x_test.shape[0]

    logger.info("Test feature data dimensions: {}".format(x_test.shape))
    logger.info("Test label data dimensions: {}".format(y_test.shape))

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        red_ball_model = LstmWithCRFModel(
            batch_size=m_args["model_args"]["batch_size"],
            n_class=m_args["model_args"]["red_n_class"],
            ball_num=m_args["model_args"]["sequence_len"] if name == "ssq" else m_args["model_args"]["red_sequence_len"],
            w_size=m_args["model_args"]["windows_size"],
            embedding_size=m_args["model_args"]["red_embedding_size"],
            words_size=m_args["model_args"]["red_n_class"],
            hidden_size=m_args["model_args"]["red_hidden_size"],
            layer_size=m_args["model_args"]["red_layer_size"]
        )
        train_step =tf.compat.v1.train.AdamOptimizer(
            learning_rate=m_args["train_args"]["red_learning_rate"],
            beta1=m_args["train_args"]["red_beta1"],
            beta2=m_args["train_args"]["red_beta2"],
            epsilon=m_args["train_args"]["red_epsilon"],
            use_locking=False,
            name='Adam'
        ).minimize(red_ball_model.loss)

        sess.run(tf.compat.v1.global_variables_initializer())
        sequence_len = m_args["model_args"]["sequence_len"] \
            if name == "ssq" else m_args["model_args"]["red_sequence_len"]
        
        for epoch in range(m_args["model_args"]["red_epochs"]):
            for i in range(train_data_len):
                _, loss_, pred = sess.run([
                    train_step, red_ball_model.loss, red_ball_model.pred_sequence
                ], feed_dict={
                    "inputs:0": x_train[i:(i+1), :, :],
                    "tag_indices:0": y_train[i:(i+1), :],
                    "sequence_length:0": np.array([sequence_len]*1)
                })
                if i % 100 == 0:
                    logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                        epoch, loss_, y_train[i:(i+1), :][0] + 1, pred[0] + 1)
                    )
        logger.info("Training time: {}".format(time.time() - start_time))
        pred_key[ball_name[0][0]] = red_ball_model.pred_sequence.name
        if not os.path.exists(m_args["path"]["red"]):
            os.makedirs(m_args["path"]["red"])
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["red"], red_ball_model_name, extension))
        logger.info("Model Evaluation【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        all_true_count = 0
        for j in range(test_data_len):
            true = y_test[j:(j + 1), :]
            pred = sess.run(red_ball_model.pred_sequence
                , feed_dict={
                    "inputs:0": x_test[j:(j + 1), :, :],
                    "sequence_length:0": np.array([sequence_len] * 1)
                })
            count = np.sum(true == pred + 1)
            all_true_count += count
            if count in eval_d:
                eval_d[count] += 1
            else:
                eval_d[count] = 1
        logger.info("Number of test sessions: {}".format(test_data_len))
        for k, v in eval_d.items():
            logger.info("Hit{} Balls，{}periods，percentage: {}%".format(k, v, round(v * 100 / test_data_len, 2)))
        logger.info(
            "Overall accuracy: {}%".format(
                round(all_true_count * 100 / (test_data_len * sequence_len), 2)
            )
        )

def train_with_eval_blue_ball_model(
        name,
        x_train,
        y_train,
        x_test,
        y_test
):
    """ Basketball model training and evaluation """
    m_args = model_args[name]
    x_train = x_train - 1 
    train_data_len = x_train.shape[0]
    if name == 'ssq':
        x_train = x_train.reshape(len(x_train),m_args["model_args"]["windows_size"])
        y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=m_args["model_args"]["blue_n_class"])
    else:
        y_train = y_train - 1
    logger.info("Training feature data dimensions: {}".format(x_train.shape))
    logger.info("Training label data dimensions: {}".format(y_train.shape))  

    x_test = x_test - 1
    test_data_len = x_test.shape[0]
    if name == "ssq":
        x_test = x_test.reshape(len(x_test), m_args["model_args"]["windows_size"])
        y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes=m_args["model_args"]["blue_n_class"])
    else:
        y_test = y_test - 1
    logger.info("Training feature data dimensions: {}".format(x_test.shape))
    logger.info("Training label data dimensions: {}".format(y_test.shape))

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        if name == "ssq":
            blue_ball_model = SignalLstmModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                outputs_size=m_args["model_args"]["blue_n_class"],
                layer_size=m_args["model_args"]["blue_layer_size"]
            )
        else:
            blue_ball_model = LstmWithCRFModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                ball_num=m_args["model_args"]["blue_sequence_len"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                words_size=m_args["model_args"]["blue_n_class"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                layer_size=m_args["model_args"]["blue_layer_size"]
            )
        train_step = tf.compat.v1.train.AdamOptimizer(
            learning_rate=m_args["train_args"]["blue_learning_rate"],
            beta1=m_args["train_args"]["blue_beta1"],
            beta2=m_args["train_args"]["blue_beta2"],
            epsilon=m_args["train_args"]["blue_epsilon"],
            use_locking=False,
            name='Adam'
        ).minimize(blue_ball_model.loss)
        sess.run(tf.compat.v1.global_variables_initializer())
        sequence_len = "" if name == "ssq" else m_args["model_args"]["blue_sequence_len"]
        for epoch in range(m_args["model_args"]["blue_epochs"]):
            for i in range(train_data_len):
                if name == "ssq":
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_label
                    ], feed_dict={
                        "inputs:0": x_train[i:(i+1), :],
                        "tag_indices:0": y_train[i:(i+1), :],
                    })
                    if i % 100 == 0:
                        logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, np.argmax(y_train[i:(i+1), :][0]) + 1, pred[0] + 1)
                        )
                else:
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_sequence
                    ], feed_dict={
                        "inputs:0": x_train[i:(i + 1), :, :],
                        "tag_indices:0": y_train[i:(i + 1), :],
                        "sequence_length:0": np.array([sequence_len] * 1)
                    })
                    if i % 100 == 0:
                        logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, y_train[i:(i + 1), :][0] + 1, pred[0] + 1)
                        )
        logger.info("Training time: {}".format(time.time() - start_time))
        pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
        if not os.path.exists(m_args["path"]["blue"]):
            os.mkdir(m_args["path"]["blue"])
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["blue"], blue_ball_model_name, extension))
        logger.info("Model Evaluation【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        all_true_count = 0
        for j in range(test_data_len):
            if name == "ssq":
                true = y_test[j:(j + 1), :]
                pred = sess.run(blue_ball_model.pred_label
                , feed_dict={"inputs:0": x_test[j:(j + 1), :]})
            else:
                true = y_test[j:(j + 1), :]
                pred = sess.run(blue_ball_model.pred_sequence
                , feed_dict={
                    "inputs:0": x_test[j:(j + 1), :, :],
                    "sequence_length:0": np.array([sequence_len] * 1)
                })
            count = np.sum(true == pred + 1)
            all_true_count += count
            if count in eval_d:
                eval_d[count] += 1
            else:
                eval_d[count] = 1
        logger.info("Number of test sessions: {}".format(test_data_len))
        for k, v in eval_d.items():
            logger.info("Hit {} balls，{} periods, percentage: {}%".format(k, v, round(v * 100 / test_data_len, 2)))
        if name == "ssq":
            logger.info(
                "Overall accuracy: {}%".format(
                    round(all_true_count * 100 / test_data_len, 2)
                )
            )
        else:
            logger.info(
                "Overall accuracy: {}%".format(
                    round(all_true_count * 100 / (test_data_len * sequence_len), 2)
                )
            )


def run(name,train_test_split):
    """ 
     Execution Training
        :param name: Gameplay
        :param train_test_split: Training set partitioning
        :return:
    """
    logger.info("Creating [{}] training set and test set...".format(name_path[name]["name"]))

    train_data, test_data = create_train_test_data(
        name,
        model_args[name]['model_args']['windows_size'],
        train_test_split
    )
    print('=======TEST DATA================')
    logger.info("Start training【{}】red ball model...".format(name_path[name]["name"]))
    train_with_eval_red_ball_model(
        name,
        x_train=train_data["red"]["x_data"], y_train=train_data["red"]["y_data"],
        x_test=test_data["red"]["x_data"], y_test=test_data["red"]["y_data"],
    )

    tf.compat.v1.reset_default_graph() #Reset Network Map

    logger.info("Start training 【{}】the basketball model...".format(name_path[name]["name"]))

    train_with_eval_blue_ball_model(
        name,
        x_train=train_data["blue"]["x_data"], y_train=train_data["blue"]["y_data"],
        x_test=test_data["blue"]["x_data"], y_test=test_data["blue"]["y_data"]
    )

    # Save the predicted key node names

    with open("{}/{}/{}".format(model_path, name, pred_key_name), "w") as f:
        json.dump(pred_key, f)

if __name__ == "__main__":
    if not args.name:
        run_name="dlt"
    else:
        run_name = args.name

    logger.info("Call training data【{}】".format(run_name))
    
    run(run_name, args.train_test_split)