from jina import Client
from docarray import Document, DocumentArray

if __name__ == '__main__':
    c = Client(host='grpc://0.0.0.0:51000')
    input_list = [
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0000.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0001.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0002.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0003.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0004.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0005.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0006.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0007.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0008.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0009.jpg"),
        Document(uri="https://github.com/YoongiKim/CIFAR-10-images/raw/master/train/airplane/0010.jpg")
    ]

    response = c.post('/query', DocumentArray(input_list), parameters={'budget': 5}).to_list()
    queried_uris = [x["uri"] for x in response]
    print(queried_uris)
