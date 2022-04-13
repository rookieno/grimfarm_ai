from ninja.files import UploadedFile
from typing import Optional, List
from ninja import File
from ninja import Schema


# 우리가 받을 형식은 파일 제목 / 이미지 파일 그 자체
class NstRequest(Schema):
    key: str
    img_url: Optional[str]
    style_url: Optional[str]