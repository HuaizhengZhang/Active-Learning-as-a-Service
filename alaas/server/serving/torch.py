import io
import torch
import torch.jit
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from .base import ServeEngine


class TorchServe(ServeEngine):
    def __init__(self, model_path):
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the latest version of a TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def batch_predict(self, inputs: torch.Tensor, is_prob=False):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        if is_prob:
            outputs = F.softmax(outputs, dim=1)

        if isinstance(outputs, tuple):
            return outputs[0].cpu().detach().numpy()
        else:
            return outputs.cpu().detach().numpy()

    # TODO: only for image model tests.
    @staticmethod
    def transform_image(image_bytes):
        t = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return t(image).unsqueeze(0)


if __name__ == '__main__':
    # TODO: only for image model tests.
    resnet_50_path = "/Users/huangyz0918/Downloads/torchscript/1.zip"
    example_image = "/Users/huangyz0918/desktop/cat.jpeg"
    with open(example_image, 'rb') as f:
        image_bytes = f.read()
        tensor = TorchServe.transform_image(image_bytes=image_bytes)
        server = TorchServe(resnet_50_path)
        prediction = server.batch_predict(tensor, is_prob=True)
        print("top, top_class: ", np.sort(prediction)[0][-1])
        print(np.shape(prediction))
