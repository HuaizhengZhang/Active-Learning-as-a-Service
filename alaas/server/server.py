import json
import os
from typing import List
from pathlib import Path

import uvicorn
import urllib.request
from urllib.parse import urlparse
from fastapi import FastAPI, File, UploadFile
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from alaas.server.util import DBManager, ConfigManager
from alaas.server.strategy import LeastConfidenceTriton
from alaas.server.util import load_image_data_as_np
from alaas.types import ALStrategyType

server_mod = FastAPI()
router = InferringRouter()


@cbv(router)
class ALServerMod:

    def __init__(self):
        self.asynchronous = False
        self.data_urls = []
        home_path = str(Path.home())
        self.alaas_home = home_path + "/.alaas/"
        self.server_config_path = self.alaas_home + "server_config.yml"
        self.db_manager = DBManager(self.alaas_home + 'index.db')

    def download_data(self, asynchronous=False):
        path_list = []
        Path(self.alaas_home).mkdir(parents=True, exist_ok=True)
        for url in self.data_urls:
            data_save_path = self.alaas_home + os.path.basename(urlparse(url).path)
            urllib.request.urlretrieve(url, data_save_path)
            path_list.append(data_save_path)
            self.db_manager.insert_record(data_save_path)
        return path_list

    def check_config(self):
        return os.path.isfile(self.server_config_path)

    @router.get("/")
    def index(self):
        if self.check_config():
            return {"message": "Welcome to ALaaS Server!",
                    "config": ConfigManager.load_config(self.server_config_path).dict()}
        else:
            return {"message": "Welcome to ALaaS Server!",
                    "config": f"Please make sure you have a server configuration at: {self.server_config_path}"}

    @router.post("/update_cfg")
    async def update_cfg(self, config: UploadFile = File(...)):
        try:
            res = await config.read()
            with open(self.server_config_path, "wb") as f:
                f.write(res)
            return {
                "status": "success",
                "message": self.server_config_path
            }
        except Exception as e:
            return {
                "message": str(e),
                "status": "failed",
            }

    @router.get("/data")
    def get_pool(self):
        return {"data": self.db_manager.read_records()}

    @router.get("/query")
    def query(self, budget: int):
        cfg_manager = ConfigManager(self.server_config_path)
        batch_size = cfg_manager.strategy.infer_model.batch_size
        model_name = cfg_manager.strategy.infer_model.name
        strategy = cfg_manager.strategy.type
        address = cfg_manager.al_server.url

        data_pool = self.db_manager.read_records()

        # TODO:  automatic data processing and data augmentation.
        results = []
        _, _, input_data = load_image_data_as_np(data_pool)
        if strategy == ALStrategyType.LEAST_CONFIDENCE:
            al_learner = LeastConfidenceTriton(source_data=input_data, model_name=model_name,
                                               batch_size=batch_size,
                                               address=address)
            results = al_learner.query(budget)
        return {"strategy": strategy, "budget": budget, "query_results": json.dumps(results.tolist())}

    @router.post("/push")
    def push(self, data: List[str], asynchronous: bool):
        self.data_urls = data
        self.asynchronous = asynchronous
        self.download_data(self.asynchronous)  # time-consuming operation
        return {'data': data, 'asynchronous': asynchronous}


class Server:
    """
    Server: Server Class for Active Learning Services.
    """

    def __init__(self, config_path):
        self.cfg_manager = ConfigManager(config_path)

    def start(self, host="0.0.0.0", port=8000, restful=True):
        if restful:
            server_mod.include_router(router)
            uvicorn.run(server_mod, host=host, port=port)
            # TODO: don't block at server starting line, continue to setup the input configuration file.
            # x = requests.post(f"http://{host}:{port}/update_cfg", data=self.cfg_manager.config.dict())
        else:
            # TODO: RPC Server
            raise NotImplementedError("gRPC server is not available right now.")


if __name__ == "__main__":
    example_config = '/Users/huangyz0918/desktop/alaas/examples/resnet_triton.yml'
    Server(example_config).start(host="0.0.0.0", port=8001)
