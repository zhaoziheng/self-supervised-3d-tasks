import cv2
import numpy as np
import albumentations as ab


def pad_to_final_size_3d(volume, w):
    dim = volume.shape[:3]
    dim = np.array(dim)

    f1 = ((((-1) * dim) + w) / 2).astype(np.int)    # 原始尺寸是384，crop成96
    f2 = ((((-1) * dim) + w) - f1).astype(np.int)   # f2=f1=[144,144,144]

    f1[f1 < 0] = 0
    f2[f2 < 0] = 0

    result = np.pad(volume, (*zip(f1, f2), (0, 0)), mode="constant", constant_values=0) # 在volume的每一轴的前后分别pad入f1和f2个pixel，把crop_size恢复成w
    return result


def pad_to_final_size_2d(image, w):
    return ab.PadIfNeeded(w, w, border_mode=cv2.BORDER_CONSTANT, value=0)(image=image)["image"]
