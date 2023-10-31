import torch


class ResNetModelActivator(torch.nn.Module):

    def __init__(self, model, layer=None, is_bcos=False):
        super().__init__()
        self.model = model
        self.layer = layer
        self.is_bcos = is_bcos
        if self.layer is not None:
            if self.is_bcos:
                self.layer_list = list(self.model[0].named_children())
            else:
                self.layer_list = list(self.model.named_children())
            assert self.layer >= 0 and self.layer < len(self.layer_list)-1

    def __call__(self, img):
        if self.layer is None:
            output = self.model(img)
            feature = img
        else:
            acts = img
            if not self.is_bcos:
                for lidx in range(len(self.layer_list)-1):
                    acts = self.layer_list[lidx][1](acts)
                    if lidx == self.layer:
                        feature = acts
                acts = acts.flatten(1)
                output = self.layer_list[-1][1](acts)
            else:
                for lidx in range(len(self.layer_list)-2):
                    acts = self.layer_list[lidx][1](acts)
                    if lidx == self.layer:
                        feature = acts
                acts = self.layer_list[-1][1](acts)
                output = self.layer_list[-2][1](acts)
                output = self.model[1](output)
                output = output.flatten(1)
        return output, feature
