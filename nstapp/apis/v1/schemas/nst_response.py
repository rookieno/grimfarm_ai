from ninja import Schema


# s3에 올라간 변환된 이미지의 url
class NstResponse(Schema):
    file_url: str