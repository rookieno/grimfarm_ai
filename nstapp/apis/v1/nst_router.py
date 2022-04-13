from typing import Optional

from django.views.decorators.csrf import csrf_exempt
from .schemas import NstResponse, NstRequest
from ninja.files import UploadedFile
from ninja import Router, File, Form
from django.http import HttpRequest
from nstapp.services.nst_service import nst_apply

router = Router()


@csrf_exempt
@router.post("/", response=NstResponse)
def nst(request: HttpRequest, nst_request: NstRequest = Form(...), style_img: UploadedFile = File(None),
        img: UploadedFile = File(None)) -> dict:
    # 서비스 함수
    # nst 변환 적용 후에 s3 업로드 해주는 서비스
    file_url = nst_apply(nst_request.key, nst_request.style_url, nst_request.img_url, style_img, img)
    return {"file_url": file_url}
