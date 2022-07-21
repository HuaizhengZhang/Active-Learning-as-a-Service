"""
The active learning executor of PyTorch models.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""

import torch
import torch.jit
import torch.nn.functional as F

import os
import warnings
import importlib
import numpy as np
from pathlib import Path
from multiprocessing.pool import ThreadPool
from jina import Executor, Document, DocumentArray, requests

from alaas.types import ALStrategyType
from alaas.server.util import DBManager
from alaas.server.preprocessor import img_transform


class TorchWorker(Executor):
    def __init__(
            self,
            model_name: str = 'resnet18',
            model_repo: str = 'pytorch/vision:v0.10.0',
            device: str = None,
            minibatch_size: int = 8,
            num_worker_preprocess: int = 4,
            data_home: str = None,
            transform=img_transform,
            strategy: str = ALStrategyType.LEAST_CONFIDENCE.value,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._strategy = strategy
        self._transform = transform
        self._minibatch_size = minibatch_size

        if data_home is None:
            self._data_home = str(Path.home()) + "/.alaas/"

        self._db_manager = DBManager(self._data_home + 'index.db')
        self._data_pool = self._db_manager.read_records()
        self._data_processed = self._db_manager.get_rows()
        self._data_not_processed = self._db_manager.get_rows(inferred=False)
        self._thread_pool = ThreadPool(processes=num_worker_preprocess)

        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        if not self._device.startswith('cuda') and (
                'OMP_NUM_THREADS' not in os.environ
                and hasattr(self.runtime_args, 'replicas')
        ):
            replicas = getattr(self.runtime_args, 'replicas', 1)
            num_threads = max(1, torch.get_num_threads() // replicas)
            if num_threads < 2:
                warnings.warn(
                    f'Too many replicas ({replicas}) vs too few threads {num_threads} may result in '
                    f'sub-optimal performance.'
                )

                # NOTE: make sure to set the threads right after the torch import,
                # and `torch.set_num_threads` always take precedence over environment variables `OMP_NUM_THREADS`.
                # For more details, please see https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
                torch.set_num_threads(max(num_threads, 1))
                torch.set_num_interop_threads(1)

        # set up the active learning (for query only) model.
        # TODO: add huggingface models.
        self._model = torch.hub.load(model_repo, model=model_name, pretrained=True)
        self._model.eval().to(self._device)

    @requests
    def query(self, docs: DocumentArray, parameters, **kwargs):
        uris = []
        with self.monitor(
                name='query_inputs',
                documentation='data download and preprocess time in seconds',
        ):
            with torch.inference_mode():
                for minibatch, uri_list, batch_data in docs.map_batch(
                        self._preproc_images,
                        batch_size=self._minibatch_size,
                        pool=self._thread_pool,
                ):
                    uris += uri_list
                    minibatch.embeddings = (
                        F.softmax(self._model(batch_data.to(self._device)), dim=1)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                    )

            _doc_list = []
            al_method = getattr(importlib.import_module('alaas.server.strategy'), self._strategy)
            al_learner = al_method(pool_size=len(uris), path_mapping=uris)

            query_results = al_learner.query(parameters['budget'], docs.embeddings)

            for result in query_results:
                _doc_list.append(Document(uri=result))
            return DocumentArray(_doc_list)

    def _preproc_images(self, docs: DocumentArray):
        with self.monitor(
                name='preprocess_images_seconds',
                documentation='images preprocess time in seconds',
        ):
            _tensor_list = []
            uri_list = []
            for data in docs:
                if data.blob:
                    data.convert_blob_to_image_tensor()
                elif data.tensor is None and data.uri:
                    uri_list.append(data.uri)
                    # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
                    data.load_uri_to_image_tensor()

                if self._transform:
                    _tensor_list.append(self._transform(data.tensor))
                else:
                    _tensor_list.append(data.tensor)
            return docs, uri_list, torch.stack(_tensor_list, dim=0)
