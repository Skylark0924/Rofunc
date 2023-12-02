import cv2
import numpy as np

color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (72, 209, 204), (238, 130, 238), (221, 160, 221), (139, 69, 19),
             (218, 112, 214), (210, 105, 30), (188, 143, 143), (119, 136, 153), (153, 50, 204), (106, 90, 205),
             (135, 206, 250), (70, 130, 180), (138, 43, 226)]


def overlay_seg_w_img(image_rgb, image_seg, alpha=0.5):
    """

    :param image_rgb: WxHx3, np.ndarray
    :param image_seg: WxH, np.ndarray
    :param alpha: Segmentation mask's transparency. float = 0.5,
    :return:
    """

    def overlay(image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.
        https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined

    image_with_masks = np.copy(image_rgb)
    masks = np.unique(image_seg)

    for mask_i in masks:
        if mask_i == 0:
            continue
        mask = np.where(image_seg == mask_i, 1, 0)
        image_with_masks = overlay(image_with_masks, mask, color=color_lst[mask_i], alpha=alpha)
    return image_with_masks
