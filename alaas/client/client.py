"""
The Client of ALaaS.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""

from urllib.parse import urlparse
from docarray import Document, DocumentArray

from alaas.client.utils import chunks


class Client:
    def __init__(self, server):
        """
        Server scheme is in the format of `scheme://netloc:port`, where
            - scheme: one of grpc, websocket, http, grpcs, websockets, https
            - netloc: the server ip address or hostname
            - port: the public port of the server
        :param server: the server URI
        """
        try:
            r = urlparse(server)
            _port = r.port
            _scheme = r.scheme
            if not _scheme:
                raise
        except:
            raise ValueError(f'{server} is not a valid scheme')

        _tls = False

        if _scheme in ('grpcs', 'https', 'wss'):
            _scheme = _scheme[:-1]
            _tls = True

        if _scheme == 'ws':
            _scheme = 'websocket'  # temp fix for the core

        if _scheme in ('grpc', 'http', 'websocket'):
            _kwargs = dict(host=r.hostname, port=_port, protocol=_scheme, tls=_tls)

            from jina import Client

            self._client = Client(**_kwargs)
            self._async_client = Client(**_kwargs, asyncio=True)
        else:
            raise ValueError(f'{server} is not a valid scheme')

    def query_by_uri(self, input_uris, budget, divide_mode=False, n_drop=None):
        """
        Query the active learner by given a list of image data uris.
        @param input_uris: the input data uris (path).
        @param budget: the querying budget.
        @param divide_mode: query the data by the batch mode, can be applied to large data.
        @param n_drop: the number of dropout, default is None (1).
        @return: queried data uris.
        """
        if divide_mode:
            # set batch_size = 1 for streaming AL settings.
            for batch in chunks(input_uris, 1):
                _temp_list = []
                for uri in batch:
                    _temp_list.append(Document(uri=uri))
                _bth_push_res = self._client.post('/push', DocumentArray(_temp_list), parameters={'save_path': None})
        else:
            _doc_list = []
            for uri in input_uris:
                _doc_list.append(Document(uri=uri))

            response = self._client.post('/query', DocumentArray(_doc_list), request_size=len(_doc_list),
                                         parameters={'budget': budget, 'n_drop': n_drop}).to_list()
            return [x["uri"] for x in response]

        return None

    def query_by_text(self, texts, budget, divide_mode=False, n_drop=None):
        """
        Query the active learner by given a list of text data uris.
        @param texts: the input data texts.
        @param budget: the querying budget.
        @param divide_mode: query the data by the batch mode, can be applied to large data.
        @param n_drop: the number of dropout, default is None (1).
        @return: queried data uris.
        """
        if divide_mode:
            # set batch_size = 1 for streaming AL settings.
            for batch in chunks(texts, 1):
                _temp_list = []
                for txt in batch:
                    _temp_list.append(Document(text=txt, mime_type='text'))
                _bth_push_res = self._client.post('/push', DocumentArray(_temp_list), parameters={'save_path': None})
        else:
            _doc_list = []
            for txt in texts:
                _doc_list.append(Document(text=txt, mime_type='text'))

            response = self._client.post('/query', DocumentArray(_doc_list), request_size=len(_doc_list),
                                         parameters={'budget': budget, 'n_drop': n_drop}).to_list()
            return [x["text"] for x in response]

        return None

    def query_by_images(self, path_list, budget, divide_mode=False, n_drop=None):
        """
        Query the active learner by given a list of text data uris.
        @param path_list: the input data images (local path).
        @param budget: the querying budget.
        @param divide_mode: query the data by the batch mode, can be applied to large data.
        @param n_drop: the number of dropout, default is None (1).
        @return: queried data uris.
        """
        if divide_mode:
            # set batch_size = 1 for streaming AL settings.
            for batch in chunks(path_list, 1):
                _temp_list = []
                for pth in batch:
                    _temp_list.append(Document(blob=open(pth, "rb").read(), mime_type='image'))
                _bth_push_res = self._client.post('/push', DocumentArray(_temp_list), parameters={'save_path': None})
        else:
            _doc_list = []
            for pth in path_list:
                _doc_list.append(Document(blob=open(pth, "rb").read(), mime_type='image'))

            response = self._client.post('/query', DocumentArray(_doc_list), request_size=len(_doc_list),
                                         parameters={'budget': budget, 'n_drop': n_drop}).to_list()
            return [x["blob"] for x in response]

        return None
