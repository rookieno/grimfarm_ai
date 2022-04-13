import pathlib
import time
from typing import Optional

from ninja.files import UploadedFile
from ninja import File
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from nstapp.apps import NstappConfig
from nstapp.utils.nst_utils import load_style, upload_tensor_img, style_select


def nst_apply(key: str, style_url: str, img_url: str, style_img: UploadedFile = File(None), img: UploadedFile = File(None)) -> str:
    # 스타일 url, 스타일 파일
    # 이미지 url, 이미지 파일
    # 저장된 스타일
    # 6가지 경우의
    if img is not None:
        style_image = style_select(style_url, style_img)

        img = Image.open(img).convert('RGB')
        content_image = tf.keras.preprocessing.image.img_to_array(img)
        content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
        content_image = tf.image.resize(content_image, (512, 512))

        stylized_image = NstappConfig.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        image_url = upload_tensor_img('rookieno', stylized_image, key)

        return image_url

    elif img is None:
        style_image = style_select(style_url, style_img)

        now = int(time.time())
        img_path = tf.keras.utils.get_file(f'{now}imgurl.jpg', f'{img_url}')
        content_image = Image.open(img_path)
        content_image = tf.keras.preprocessing.image.img_to_array(content_image)
        content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
        content_image = tf.image.resize(content_image, (512, 512))

        stylized_image = NstappConfig.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        image_url = upload_tensor_img('rookieno', stylized_image, key)

        return image_url

