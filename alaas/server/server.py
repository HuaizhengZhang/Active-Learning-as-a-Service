import os
from typing import List
from pathlib import Path

import uvicorn
import urllib.request
from urllib.parse import urlparse
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from alaas.server.util import ConfigManager
from alaas.server.util import DBManager

EXAMPLE_CONFIG_PATH = '/Users/huangyz0918/desktop/alaas/examples/resnet_triton.yml'

app_server = FastAPI()
router = InferringRouter()


@cbv(router)
class ALServerMod:

    def __init__(self):
        self.asynchronous = False
        self.data_urls = []
        home_path = str(Path.home())
        self.alaas_home = home_path + "/.alaas/"
        self.db_manager = DBManager(self.alaas_home + 'index.db')

    def download_data(self):
        path_list = []
        Path(self.alaas_home).mkdir(parents=True, exist_ok=True)
        for url in self.data_urls:
            data_save_path = self.alaas_home + os.path.basename(urlparse(url).path)
            urllib.request.urlretrieve(url, data_save_path)
            path_list.append(data_save_path)
            self.db_manager.insert_record(data_save_path)
        return path_list

    @router.get("/")
    def index(self):
        return {"message": "Welcome to ALaaS Server!"}

    @router.get("/data")
    def get_pool(self):
        return {"data": self.db_manager.read_records()}

    @router.get("/query")
    def query(self, budget: int):
        return {"budget": budget}

    @router.post("/push")
    def push(self, data: List[str], asynchronous: bool):
        self.data_urls = data
        self.asynchronous = asynchronous
        self.download_data()  # time-consuming operation
        return {'data': data, 'asynchronous': asynchronous}


class Server:
    """
    Server: Server Class for Active Learning Services.
    """

    def __init__(self, config_path):
        self.cfg_manager = ConfigManager(config_path)

    @staticmethod
    def start(host="0.0.0.0", port=8000, restful=True):
        if restful:
            app_server.include_router(router)
            uvicorn.run(app_server, host=host, port=port)
        else:
            # TODO: RPC Server
            raise NotImplementedError("gRPC server is not available right now.")


if __name__ == "__main__":
    Server(config_path=EXAMPLE_CONFIG_PATH).start(host="0.0.0.0", port=8001)
