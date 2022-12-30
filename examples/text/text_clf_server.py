"""
Server example for text classification active learning tasks.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 23/07/2022
"""

from alaas.server import Server

SERVER_CONFIG = './distilbert_base.yml'

if __name__ == '__main__':
    # start the server by an input configuration file.
    # Server.start_by_config(SERVER_CONFIG)

    Server.start(model_hub="huggingface/pytorch-transformers", 
                    model_name="distilbert-base-uncased",
                    tokenizer="distilbert-base-uncased",
                    transformers_task="text-classification",
                    strategy="RandomSampling"
                )
