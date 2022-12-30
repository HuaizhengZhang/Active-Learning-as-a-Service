"""
The active learning executor of PyTorch models.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""
import copy

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

from alaas.server.util import DBManager
from alaas.types import ALStrategyType, ModalityType
from alaas.server.preprocessor import img_transform


class TorchALWorker(Executor):
    """
    TorchALWorker: the backend worker class of PyTorch models.
    """

    def __init__(
            self,
            model_name: str = 'resnet18',
            model_repo: str = 'pytorch/vision:v0.10.0',
            device: str = None,
            minibatch_size: int = 8,
            num_worker_preprocess: int = 4,
            data_home: str = None,
            tokenizer_model: str = None,
            task: str = None,
            transform=None,
            strategy: str = ALStrategyType.LEAST_CONFIDENCE.value,
            *args,
            **kwargs,
    ):
        """
        The parameters of build a TorchWorker.
        @param model_name: the active learning model name (default is resnet18).
        @param model_repo: the model hub name, for example: pytorch/vision:v0.10.0, models will be downloaded from here.
        @param device: the model running device (default is 'cpu').
        @param minibatch_size: the mini batch size for data processing.
        @param num_worker_preprocess: the number of backend workers (multi-thread).
        @param data_home: the data home path for storing the unlabeled data.
        @param tokenizer_model: [for nlp tasks], the model name of tokenizer, for example, 'bert-large-uncased'.
        @param task: [for nlp tasks], the task name of transformers pipeline, for example, 'text-classification'.
        @param transform: the data transform for pre-processing.
        @param strategy: the active learning strategy (for example, 'LeastConfidence').
        @param args: the args for backend server.
        @param kwargs: the kwargs for backend server.
        """
        super().__init__(*args, **kwargs)

        self._strategy = strategy
        self._transform = transform
        self._minibatch_size = minibatch_size
        self._thread_pool = ThreadPool(processes=num_worker_preprocess)

        self._task = task
        self._model_repo = model_repo
        self._model_name = model_name
        self._tokenizer_model = tokenizer_model

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
        
        # skip loading models while using random sampling strategy.
        if self._strategy != ALStrategyType.RANDOM_SAMPLING.value:
            self._init_model()
        else:
            print("\n skip initializing deep learning models for random sampling strategy. \n")

    def _init_model(self):
        # set up the active learning (for query only) model.
        if self._tokenizer_model and 'huggingface' in self._model_repo:
            self._data_modality = ModalityType.TEXT
            from transformers import AutoTokenizer, pipeline
            _tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_model)
            # build task-specify parameters (TODO: adapt to more tasks).
            hf_extra_parameters = {}
            if self._task == 'text-classification':
                hf_extra_parameters.update({'return_all_scores': True})
            # build the pipeline.
            self._model = pipeline(self._task,
                                model=self._model_name,
                                tokenizer=_tokenizer,
                                device=self._convert_torch_device(),
                                **hf_extra_parameters)
        else:
            self._data_modality = ModalityType.IMAGE
            self._model = torch.hub.load(self._model_repo, model=self._model_name, pretrained=True)
            self._model.eval().to(self._device)

    def _convert_torch_device(self):
        """
        converting the pytorch devices to the format of transformers.
        @return: the processed device.
        """
        if self._device == 'cpu':
            return -1
        elif self._device == 'cuda':
            return 0
        else:
            return self._device.split(':')[-1]

    def _set_input_modality(self, sample: Document):
        """
        Check and set the global data modality for the worker.
        @param sample: one of the sample from all the unlabeled data samples.
        @return: None.
        """
        if sample.blob or sample.uri:
            self._data_modality = ModalityType.IMAGE
        elif sample.text:
            self._data_modality = ModalityType.TEXT
        else:
            raise TypeError("unsupported data modality, data should be neither image or text.")

    def _get_model_embeddings(self, docs: DocumentArray):
        """
        Get the processed data embedding by given deep learning models.
        @param docs: the input data in DocumentArray.
        @return: the return values in DocumentArray with embeddings.
        """
        index_pths = []
        for minibatch, index_list, batch_data in docs.map_batch(
                self._preproc_data,
                batch_size=self._minibatch_size,
                pool=self._thread_pool,
        ):
            index_pths += index_list
            if self._strategy != ALStrategyType.RANDOM_SAMPLING.value:
                if self._data_modality == ModalityType.IMAGE:
                    minibatch.embeddings = (
                        F.softmax(self._model(batch_data.to(self._device)), dim=1)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                    )
                elif self._data_modality == ModalityType.TEXT:
                    results = []
                    for item in self._model(batch_data):
                        results.append([x['score'] for x in item])
                    minibatch.embeddings = (
                        np.array(results)
                    )
        return index_pths, docs

    @requests(on='/push')
    def push(self, docs: DocumentArray, parameters, **kwargs):
        """
        Active learning push function. This function is designed for large-scale input data/uris,
        the data will be fetched first and store temporary in the centre AL server.
        @param docs: the uploaded/targeted data, can be blob/uri for CV tasks or text for NLP tasks.
        @param parameters: the parameters include the active learning budget, etc.
        @param kwargs: the kwargs for the backend server.
        @return: the queried data in uris/blobs/texts.
        """
        index_pths = []

        # save the docs into local files.
        if parameters['save_path']:
            save_path = parameters['save_path']
        else:
            save_path = self._data_home

        for doc in docs:
            data_path = save_path + doc.id
            index_pths.append(Document(content=data_path))

            if doc.blob:
                doc.save_blob_to_file(data_path)

            elif doc.uri:
                doc.load_uri_to_blob()
                doc.save_blob_to_file(data_path)

            elif doc.text:
                doc.convert_text_to_datauri()
                doc.save_uri_to_file(data_path)

            self._db_manager.insert_record(data_path)

        return DocumentArray(index_pths)

    @requests(on='/query')
    def query(self, docs: DocumentArray, parameters, **kwargs):
        """
        Active learning query function. This function is an end-to-end data query function,
        the data need to be uploaded/downloaded after calling the function.
        @param docs: the uploaded/targeted data, can be blob/uri for CV tasks or text for NLP tasks.
        @param parameters: the parameters include the active learning budget, etc.
            @param budget: the active learning query budget, required for each query.
            @param n_drop: the number of dropout, default is None.
        @param kwargs: the kwargs for the backend server.
        @return: the queried data in uris/blobs/texts.
        """
        index_pths = []
        input_embeddings = None
        with self.monitor(
                name='query_inputs',
                documentation='data download and preprocess time in seconds',
        ):
            with torch.inference_mode():
                if 'n_drop' in parameters.keys() and parameters['n_drop'] is not None:
                    for _ in range(int(parameters['n_drop'])):
                        index_pths, round_docs = self._get_model_embeddings(copy.deepcopy(docs))
                        if input_embeddings is None:
                            input_embeddings = round_docs.embeddings
                            input_embeddings = np.stack((input_embeddings, round_docs.embeddings))
                        else:
                            input_embeddings = np.concatenate((input_embeddings, [round_docs.embeddings]), axis=0)
                else:
                    index_pths, docs = self._get_model_embeddings(docs)
                    input_embeddings = docs.embeddings

            _doc_list = []
            al_method = getattr(importlib.import_module('alaas.server.strategy'), self._strategy)
            if 'n_drop' in parameters.keys() and parameters['n_drop'] is not None:
                al_learner = al_method(path_mapping=index_pths, n_drop=int(parameters['n_drop']))
            else:
                al_learner = al_method(path_mapping=index_pths, n_drop=None)
            query_results = al_learner.query(parameters['budget'], input_embeddings)

            for result in query_results:
                if self._data_modality == ModalityType.IMAGE:
                    _doc_list.append(Document(blob=result.blob, uri=result.uri))

                elif self._data_modality == ModalityType.TEXT:
                    _doc_list.append(Document(text=result.text, uri=result.uri))

            return DocumentArray(_doc_list)

    def _preproc_data(self, docs: DocumentArray):
        """
        pre-processing function for the active learning input data.
        @param docs: the input data.
        @return: original data objects, data index list (for active learning mapping), and the processed data.
        """
        with self.monitor(
                name='preprocess_data_seconds',
                documentation='images preprocess time in seconds',
        ):
            _tensor_list = []
            index_list = []

            self._set_input_modality(docs[0])

            if self._data_modality == ModalityType.IMAGE:
                self._transform = img_transform
                for data in docs:
                    if data.blob:
                        index_doc = Document(blob=data.blob)
                        data.convert_blob_to_image_tensor()
                    elif data.tensor is None and data.uri:
                        # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
                        index_doc = Document(uri=data.uri)
                        data.load_uri_to_image_tensor()

                    index_list.append(index_doc)
                    _tensor_list.append(self._transform(data.tensor))
                return_outputs = torch.stack(_tensor_list, dim=0)
            elif self._data_modality == ModalityType.TEXT:
                for data in docs:
                    if self._tokenizer_model:
                        index_list.append(Document(text=data.text))
                        # TODO: text data pre-processing here.
                        _tensor_list.append(data.text)
                        return_outputs = _tensor_list
                    else:
                        raise ValueError(
                            """
                            For text data, you need to set the tokenizer model in your 
                            configuration file (e.g., tokenizer: \"distilbert-base-uncased\") 
                            under the \"active_learning.strategy.model\"
                            """)

            return docs, index_list, return_outputs
