import os
ball_name=[
   ("Num","red"),
   ("NumS","blue")
]

data_file_name = "power.csv"
data_mega_file_name='mega.csv'
name_path = {
    "ssq":{
        "name":"Power6/55",
        "path": "data/results/"
    },
    "dlt": {
        "name": "Power",
        "path": "data/results/"
    },
    "mega": {
        "name": "Mega",
        "path": "data/results/"
    }
}
model_path = os.getcwd() + "/model/"

model_args={
    "ssq":{
        "model_args":{
            "windows_size":12,
            "batch_size": 1,
            "sequence_len": 5,
            "red_n_class": 45,
            "red_epochs": 1,
            "red_embedding_size": 45,
            "red_hidden_size": 45,
            "red_layer_size": 1,
            "blue_n_class": 45,
            "blue_epochs": 1,
            "blue_embedding_size": 45,
            "blue_hidden_size": 45,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/ssq/red_ball_model/",
            "blue": model_path + "/ssq/blue_ball_model/"
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 6,
            "red_n_class": 56,
            "red_epochs": 1,
            "red_embedding_size": 320,
            "red_hidden_size": 16,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 55,
            "blue_epochs": 1,
            "blue_embedding_size": 55,
            "blue_hidden_size": 55,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/dlt/red_ball_model/",
            "blue": model_path + "/dlt/blue_ball_model/"
        }
    },
    "mega": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "ball_sequence_len": 6,
            "ball_n_class": 45,
            "ball_epochs": 1,
            "ball_embedding_size": 320,
            "ball_hidden_size": 16,
            "ball_layer_size": 1,
        },
        "train_args": {
            "ball_learning_rate": 0.001,
            "ball_beta1": 0.9,
            "ball_beta2": 0.999,
            "ball_epsilon": 1e-08,
        },
        "path": {
            "ball": model_path + "/mega/",
        }
    }
}

pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
mega_model_name = "mega_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"