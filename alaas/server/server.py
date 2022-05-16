import uvicorn
from typing import List

import os
from pathlib import Path
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
class ALServerModel:

    def __init__(self):
        self.asynchronous = False
        self.data_urls = []

    def download_data(self):
        home_path = str(Path.home())
        alaas_home = home_path + "/.alaas/"
        Path(alaas_home).mkdir(parents=True, exist_ok=True)
        for url in self.data_urls:
            urllib.request.urlretrieve(url, alaas_home + os.path.basename(urlparse(url).path))

    @router.get("/")
    def index(self):
        return {"message": "Welcome to ALaaS Server!"}

    @router.get("/query")
    def query(self, budget: int):
        return {"budget": budget}

    @router.post("/push")
    def push(self, data: List[str], asynchronous: bool):
        self.data_urls = data
        self.asynchronous = asynchronous
        self.download_data()
        return {'data': data, 'asynchronous': asynchronous}


class Server:
    """
    Server: Server Class for Active Learning Services.
    """

    def __init__(self, config_path):
        self.cfg_manager = ConfigManager(config_path)
        self.db_manager = None

    def start_db(self, db_name, container_name, ports):
        # TODO: check the port and db name is valid or not, offer some options to start.
        self.cfg_manager = DBManager(db_name, container_name, ports)

    @staticmethod
    def start(host="0.0.0.0", port=8000, restful=True):
        if restful:
            app_server.include_router(router)
            uvicorn.run(app_server, host=host, port=port)
        else:
            # RPC Server
            raise NotImplementedError("gRPC server is not available right now.")


if __name__ == "__main__":
    Server(config_path=EXAMPLE_CONFIG_PATH).start(host="0.0.0.0", port=8001)
