import os

import numpy as np
import pip
import torch
from matplotlib import pyplot as plt

from rofunc.learning.pre_trained_models.download import download_ckpt
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.utils.oslab import get_rofunc_path
from rofunc.utils.visualab.image import show_anns, show_mask, show_star_points_w_labels
from rofunc.utils.visualab.interact import mouse_click_coords, mouse_select_rec_region
from torchvision.transforms import ToTensor


def efficient_sam_generate(image,
                           efficient_sam_checkpoint="efficientsam_s_gpu.jit",
                           model_type="s"):
    ckpt_path = os.path.join(get_rofunc_path(), "learning/pre_trained_models", efficient_sam_checkpoint)
    if not os.path.exists(ckpt_path):
        if model_type == "s":
            download_ckpt(
                f"https://www.dropbox.com/scl/fi/ziif8xudwbyyphb4tohza/efficientsam_s_gpu.jit?rlkey=8aflq9kf0bfujz5ex4lxuoq56&dl=0",
                efficient_sam_checkpoint)
        elif model_type == "ti":
            download_ckpt(
                f"https://www.dropbox.com/scl/fi/lup5s4gthmlv6qf3f5zz3/efficientsam_ti_gpu.jit?rlkey=pap1xktxw50qiaey17no16bqz&dl=0",
                efficient_sam_checkpoint)
        else:
            raise ValueError("model_type should be either s or ti")

    img_tensor = ToTensor()(image)

    model = torch.jit.load(ckpt_path)
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()
    
    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(selected_mask_using_predicted_iou)
    plt.axis('off')
    plt.show()


def efficient_sam_predict(image,
                          use_point=False,
                          use_box=False,
                          choose_best_mask=True,
                          efficient_sam_checkpoint="efficientsam_s_gpu.jit",
                          model_type="s"):
    """
    Use mouse to select points or a box, and segment the object with prompt in the image.

    :param image: image to segment
    :param use_point: whether to use pos/neg points to segment
    :param use_box: whether to use a box to segment
    :param choose_best_mask: whether to only choose the best mask
    :param efficient_sam_checkpoint: checkpoint path of sam model
    :param model_type: sam model type
    :return:
    """
    assert use_point != use_box, "Either use_point or use_box should be True"

    ckpt_path = os.path.join(get_rofunc_path(), "learning/pre_trained_models", efficient_sam_checkpoint)
    if not os.path.exists(ckpt_path):
        if model_type == "s":
            download_ckpt(
                f"https://www.dropbox.com/scl/fi/ziif8xudwbyyphb4tohza/efficientsam_s_gpu.jit?rlkey=8aflq9kf0bfujz5ex4lxuoq56&dl=0",
                efficient_sam_checkpoint)
        elif model_type == "ti":
            download_ckpt(
                f"https://www.dropbox.com/scl/fi/lup5s4gthmlv6qf3f5zz3/efficientsam_ti_gpu.jit?rlkey=pap1xktxw50qiaey17no16bqz&dl=0",
                efficient_sam_checkpoint)
        else:
            raise ValueError("model_type should be either s or ti")

    beauty_print("Segment with prompt", type="module")

    img_tensor = ToTensor()(image)

    model = torch.jit.load(ckpt_path)

    if use_point and not use_box:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plt.imshow(image)
        pos_input_point = mouse_click_coords(fig, ax, "positive")
        plt.axis('off')
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plt.imshow(image)
        neg_input_point = mouse_click_coords(fig, ax, "negative")
        plt.axis('off')
        plt.show()

        input_point = np.concatenate([pos_input_point, neg_input_point], axis=0)
        input_point_tensor = torch.reshape(torch.tensor(input_point), [1, 1, -1, 2])
        input_label = np.concatenate([np.ones(len(pos_input_point)), np.zeros(len(neg_input_point))], axis=0)
        input_label_tensor = torch.reshape(torch.tensor(input_label), [1, 1, -1])
        predicted_logits, predicted_iou = model(
            img_tensor[None, ...].cuda(),
            input_point_tensor.cuda(),
            input_label_tensor.cuda(),
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()
    elif not use_point and use_box:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plt.imshow(image)
        input_box = mouse_select_rec_region(fig, ax)
        plt.axis('off')
        plt.show()

        input_box_tensor = torch.reshape(torch.tensor(input_box), [1, 1, 2, 2])
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
        predicted_logits, predicted_iou = model(
            img_tensor[None, ...].cuda(),
            input_box_tensor.cuda(),
            bbox_labels.cuda(),
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()
    else:
        raise ValueError("Either use_point or use_box should be True")

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(selected_mask_using_predicted_iou, plt.gca())
    if use_point:
        show_star_points_w_labels(input_point, input_label, plt.gca())
    plt.title(f"EfficientSAM", fontsize=18)
    plt.axis('off')
    plt.show()
