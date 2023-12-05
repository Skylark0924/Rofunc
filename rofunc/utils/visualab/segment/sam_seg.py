import os

import numpy as np
import pip
from matplotlib import pyplot as plt

from rofunc.learning.pre_trained_models.download import download_ckpt
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.utils.oslab import get_rofunc_path
from rofunc.utils.visualab.image import show_anns, show_mask, show_star_points_w_labels
from rofunc.utils.visualab.interact import mouse_click_coords, mouse_select_rec_region


def sam_generate(image,
                 sam_checkpoint="sam_vit_h_4b8939.pth",
                 model_type="vit_h"):
    try:
        import segment_anything
    except ImportError:
        print("segment-anything is not installed. Install it automatically...")
        pip.main(['install', 'git+https://github.com/facebookresearch/segment-anything.git'])
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    ckpt_path = os.path.join(get_rofunc_path(), "learning/pre_trained_models", sam_checkpoint)
    if not os.path.exists(ckpt_path):
        download_ckpt(f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}", sam_checkpoint)

    sam = sam_model_registry[model_type](checkpoint=ckpt_path).to(device="cuda:0")
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=32,
                                               pred_iou_thresh=0.86,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100,  # Requires open-cv to run post-processing
                                               )
    masks = mask_generator.generate(image)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()


def sam_predict(image,
                use_point=False,
                use_box=False,
                choose_best_mask=True,
                sam_checkpoint="sam_vit_h_4b8939.pth",
                model_type="vit_h"):
    """
    Use mouse to select points or a box, and segment the object with prompt in the image.

    :param image: image to segment
    :param use_point: whether to use pos/neg points to segment
    :param use_box: whether to use a box to segment
    :param choose_best_mask: whether to only choose the best mask
    :param sam_checkpoint: checkpoint path of sam model
    :param model_type: sam model type
    :return:
    """
    try:
        import segment_anything
    except ImportError:
        print("segment-anything is not installed. Install it automatically...")
        pip.main(['install', 'git+https://github.com/facebookresearch/segment-anything.git'])
    from segment_anything import SamPredictor, sam_model_registry

    assert use_point != use_box, "Either use_point or use_box should be True"
    ckpt_path = os.path.join(get_rofunc_path(), "learning/pre_trained_models", sam_checkpoint)
    if not os.path.exists(ckpt_path):
        download_ckpt(f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}", sam_checkpoint)

    beauty_print("Segment with prompt", type="module")

    sam = sam_model_registry[model_type](checkpoint=ckpt_path).to(device="cuda:0")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

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
        input_label = np.concatenate([np.ones(len(pos_input_point)), np.zeros(len(neg_input_point))], axis=0)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=not choose_best_mask,
        )
    elif not use_point and use_box:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plt.imshow(image)
        input_box = mouse_select_rec_region(fig, ax)
        plt.axis('off')
        plt.show()

        masks, scores, logits = predictor.predict(
            box=np.array(input_box),
            multimask_output=not choose_best_mask,
        )
    else:
        raise ValueError("Either use_point or use_box should be True")

    if not choose_best_mask:
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            if use_point:
                show_star_points_w_labels(input_point, input_label, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        if use_point:
            show_star_points_w_labels(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show()
