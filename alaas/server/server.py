import hashlib
import os
import json
import importlib
from threading import Thread

import numpy as np
from typing import List
from pathlib import Path

import uvicorn
from urllib.parse import urlparse
from fastapi import FastAPI, File, UploadFile
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from alaas.types import ALStrategyType
from alaas.server.util import DBManager, ConfigManager
from alaas.server.util import load_images_data_as_np, load_image_data_as_np
from alaas.server.serving import triton_inference_func
from alaas.server.util import UrlDownloader

server_mod = FastAPI()
router = InferringRouter()


class AsyncUrlProc(Thread):
    """
    Async Data Downloader and Processor.
    """

    def __init__(self, data_url, model_name, server_url, inference_func=triton_inference_func,
                 proc_func=load_image_data_as_np):
        Thread.__init__(self)
        self.inference_func = inference_func
        self.proc_func = proc_func
        self.model_name = model_name
        self.server_addr = server_url
        self.data_url = data_url
        home_path = str(Path.home())
        self.alaas_home = home_path + "/.alaas/"
        self.db_manager = DBManager(self.alaas_home + 'index.db')
        self._return = None

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

    def run(self):
        data_save_path = self.alaas_home + os.path.basename(urlparse(self.data_url).path)
        UrlDownloader().download(data_save_path, self.data_url)
        inference_result = np.array(self.inference_func(
            np.array(self.proc_func(data_save_path), dtype=np.float32), 1,
            model_name=self.model_name, address=self.server_addr
        ))
        file_id = hashlib.md5(data_save_path.encode('utf-8')).hexdigest()
        if self.db_manager.check_row(file_id):
            self.db_manager.update_inference_md5(file_id, inference_result)
        else:
            self.db_manager.insert_record(data_save_path, inference_result)
        self._return = data_save_path


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
        cfg_manager = ConfigManager(self.server_config_path)
        model_name = cfg_manager.strategy.infer_model.name
        address = cfg_manager.al_server.url
        Path(self.alaas_home).mkdir(parents=True, exist_ok=True)
        proc_threads = []
        for url in self.data_urls:
            if asynchronous:
                processor = AsyncUrlProc(data_url=url, model_name=model_name, server_url=address,
                                         inference_func=triton_inference_func, proc_func=load_image_data_as_np)
                processor.start()
                proc_threads.append(processor)
            else:
                data_save_path = self.alaas_home + os.path.basename(urlparse(url).path)
                UrlDownloader().download(data_save_path, url)
                path_list.append(data_save_path)
                self.db_manager.insert_record(data_save_path, None)
        for thread in proc_threads:
            path_list.append(thread.join())
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

        try:
            al_method = getattr(importlib.import_module('alaas.server.strategy'), strategy.value)
            if strategy == ALStrategyType.RANDOM_SAMPLING:
                al_learner = al_method()
            else:
                al_learner = al_method(infer_func=triton_inference_func,
                                       proc_func=load_images_data_as_np,
                                       model_name=model_name,
                                       batch_size=batch_size,
                                       address=address)
            results = al_learner.query(budget)
            return {"strategy": strategy, "budget": budget, "query_results": json.dumps(results.tolist())}
        except Exception as e:
            return {
                "message": str(e),
                "status": "failed",
            }

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
    example_config = '/Users/huangyz0918/desktop/alaas/examples/config.yml'
    Server(example_config).start(host="0.0.0.0", port=8001)
