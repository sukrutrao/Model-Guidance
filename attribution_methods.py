import torch
import captum
import captum.attr


def get_attributor(model, attributor_name, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
    attributor_map = {
        "BCos": BCosAttributor,
        "GradCam": GradCamAttributor,
        "IxG": IxGAttributor
    }
    return attributor_map[attributor_name](model, only_positive, binarize, interpolate, interpolate_dims, batch_mode)


class AttributorBase:

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__()
        self.model = model
        self.only_positive = only_positive
        self.binarize = binarize
        self.interpolate = interpolate
        self.interpolate_dims = interpolate_dims
        self.batch_mode = batch_mode

    def __call__(self, feature, output, class_idx=None, img_idx=None, classes=None):
        if self.batch_mode:
            return self._call_batch_mode(feature, output, classes)
        return self._call_single(feature, output, class_idx, img_idx)

    def _call_batch_mode(self, feature, output, classes):
        raise NotImplementedError

    def _call_single(self, feature, output, class_idx, img_idx):
        raise NotImplementedError

    def check_interpolate(self, attributions):
        if self.interpolate:
            return captum.attr.LayerAttribution.interpolate(
                attributions, interpolate_dims=self.interpolate_dims, interpolate_mode="bilinear")
        return attributions

    def check_binarize(self, attributions):
        if self.binarize:
            attr_max = attributions.abs().amax(dim=(1, 2, 3), keepdim=True)
            attributions = torch.where(
                attr_max == 0, attributions, attributions/attr_max)
        return attributions

    def check_only_positive(self, attributions):
        if self.only_positive:
            return attributions.clamp(min=0)
        return attributions

    def apply_post_processing(self, attributions):
        attributions = self.check_only_positive(attributions)
        attributions = self.check_binarize(attributions)
        attributions = self.check_interpolate(attributions)
        return attributions


class BCosAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        with self.model.explanation_mode():
            grads = torch.autograd.grad(torch.unbind(
                target_outputs), feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads*feature).sum(dim=1, keepdim=True)
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx):
        with self.model.explanation_mode():
            grads = torch.autograd.grad(
                output[img_idx, class_idx], feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads[img_idx]*feature[img_idx]
                        ).sum(dim=0, keepdim=True).unsqueeze(0)
        return self.apply_post_processing(attributions)


class GradCamAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(torch.unbind(
            target_outputs), feature, create_graph=True, retain_graph=True)[0]
        grads = grads.mean(dim=(2, 3), keepdim=True)
        prods = grads * feature
        attributions = torch.nn.functional.relu(
            torch.sum(prods, axis=1, keepdim=True))
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx):
        grads = torch.autograd.grad(
            output[img_idx, class_idx], feature, create_graph=True, retain_graph=True)[0]
        grads = grads.mean(dim=(2, 3), keepdim=True)
        prods = grads[img_idx] * feature[img_idx]
        attributions = torch.nn.functional.relu(
            torch.sum(prods, axis=0, keepdim=True)).unsqueeze(0)
        return self.apply_post_processing(attributions)


class IxGAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(torch.unbind(
            target_outputs), feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads * feature).sum(dim=1, keepdim=True)
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx):
        grads = torch.autograd.grad(
            output[img_idx, class_idx], feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads[img_idx] * feature[img_idx]
                        ).sum(dim=0, keepdim=True).unsqueeze(0)
        return self.apply_post_processing(attributions)
