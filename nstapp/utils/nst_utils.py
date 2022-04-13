import tensorflow as tf
from nstapp.apps import NstappConfig
import numpy as np
from PIL import Image
from io import BytesIO
import time


def load_style(path_to_style, max_dim):
    img = tf.io.read_file(path_to_style)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 전체 이미지를 비율을 유지하면서, 우리가 원하는 크기로 줄이기
    # 200 x 400 x 3
    # 200 / 400 = 1/2
    shape = tf.cast(tf.shape(img)[:-1], tf.float32) # 200 x 200 만 추출 차원 부분은 빼고
    long_dim = max(shape)
    scale = max_dim / long_dim

    # 200 x 400 소수점이 있으면 안됨 픽셀이기 때문에
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :] # 200 x 400 x 3 -> 10 x 200 x 400 x 3
    return img


# 정규화된 이미지 헤제
def upload_tensor_img(bucket, tensor, key):
    tensor = np.array(tensor*255, dtype=np.uint8)
    # image 화
    image = Image.fromarray(tensor[0])
    buffer = BytesIO()
    image.save(buffer, 'PNG') # 메모리에 파일을 잠시 저장해놓는다
    buffer.seek(0) # 메모리 전체를 읽기위해서 0번째 포인터부터 파일을 읽음
    # s3 업로드
    NstappConfig.s3.put_object(Bucket=bucket, Key=key, Body=buffer, ACL='public-read')
    # 업로드 링크 리턴
    location = NstappConfig.s3.get_bucket_location(Bucket=bucket)['LocationConstraint']
    url = "https://s3-%s.amazonaws.com/%s/%s" % (location, bucket, key)
    return url


def style_select(style_url, style_img):
    if style_img is not None:
        style_image = Image.open(style_img).convert('RGB')
        style_image = tf.keras.preprocessing.image.img_to_array(style_image)
        style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
        style_image = tf.image.resize(style_image, (512, 512))
    elif style_img is None and style_url != '':
        now = int(time.time())
        style_path = tf.keras.utils.get_file(f'{now}styleurl.jpg', f'{style_url}')
        style_image = load_style(style_path, 512)

    return style_image
