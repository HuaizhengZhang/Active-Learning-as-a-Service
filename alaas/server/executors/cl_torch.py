"""
The active learning model updater of PyTorch models.
@author huangyz0918 (huangyz0918@gmail.com)
@date 10/08/2022
"""
import torch
import torch.jit

import os
import warnings
from pathlib import Path
from multiprocessing.pool import ThreadPool
from jina import Executor, DocumentArray, requests

from alaas.server.util import DBManager


class TorchCLWorker(Executor):
    """
    TorchCLWorker: the backend model updating worker class of PyTorch models.
    """

    def __init__(
            self,
            device: str = None,
            minibatch_size: int = 128,
            num_worker_preprocess: int = 4,
            data_home: str = None,
            transform=None,
            *args,
            **kwargs,
    ):
        """
        TODO: add river-torch (https://github.com/online-ml/river-torch) or implement an online learning module here.
        """
        super().__init__(*args, **kwargs)

        self._transform = transform
        self._minibatch_size = minibatch_size
        self._thread_pool = ThreadPool(processes=num_worker_preprocess)

        if data_home is None:
            self._data_home = str(Path.home()) + "/.alaas/"
            Path(self._data_home).mkdir(parents=True, exist_ok=True)

        self._db_manager = DBManager(self._data_home + 'cache.db')
        self._data_pool = self._db_manager.read_records()
        self._data_processed = self._db_manager.get_rows()
        self._data_not_processed = self._db_manager.get_rows(inferred=False)

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

                torch.set_num_threads(max(num_threads, 1))
                torch.set_num_interop_threads(1)

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters, **kwargs):
        """
        Active learning push function. This function is designed for large-scale input data/uris,
        the data will be fetched first and store temporary in the centre AL server.
        @param docs: the uploaded/targeted data, can be blob/uri for CV tasks or text for NLP tasks.
        @param parameters: the parameters include the active learning budget, etc.
        @param kwargs: the kwargs for the backend server.
        @return: the queried data in uris/blobs/texts.
        """
        index_pths = []
        # TODO: update the active learner at scale.
        return DocumentArray(index_pths)
