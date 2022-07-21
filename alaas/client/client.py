"""
The Client of ALaaS.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""

from urllib.parse import urlparse
from docarray import Document, DocumentArray


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

    def query_by_uri(self, input_uris, budget):
        """
        Query the active learner by given a list of data uris.
        @param input_uris: the input data uris (path).
        @param budget: the querying budget.
        @return: queried data uris.
        """
        _doc_list = []
        for uri in input_uris:
            _doc_list.append(Document(uri=uri))

        response = self._client.post('/query', DocumentArray(_doc_list), parameters={'budget': budget}).to_list()
        return [x["uri"] for x in response]
