import numpy as np
import cv2
import torch


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            # 正向传播 hook
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            # 反向传播 hook
            self.handles.append(
                target_layer.register_full_backward_hook(self.save_gradient)
            )

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    @staticmethod
    def get_loss(output, target_category):
        """支持单张图片和 batch"""
        if isinstance(target_category, int):
            return output[0, target_category]
        elif isinstance(target_category, (list, tuple, np.ndarray)):
            loss = 0
            for i in range(len(target_category)):
                loss += output[i, target_category[i]]
            return loss
        else:
            raise ValueError("target_category must be int or list/tuple/ndarray")

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    @staticmethod
    def scale_cam_image(cam_list, target_size=None):
        """ cam_list 是 list，返回 list """
        result = []
        for cam in cam_list:
            img = cam - np.min(cam)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        return result

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0
            # 保证二维 (H, W)
            if cam.ndim == 3:
                cam = cam[0]
            scaled = self.scale_cam_image([cam], target_size=target_size)[0]
            cam_per_target_layer.append(scaled)
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_layer):
        """将多个 layer 的 CAM 做平均融合"""
        return np.mean(cam_per_layer, axis=0)

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:

    # mask 确保二维
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, but got shape {mask.shape}")

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = np.float32(heatmap) / 255.0
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
