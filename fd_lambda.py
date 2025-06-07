import json
import base64
import boto3
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN
import numpy as np

sqs = boto3.client(
    'sqs',
    region_name='us-east-1', 
    aws_access_key_id="",
    aws_secret_access_key=""
)
REQUEST_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/430118847891/11111-req-queue'
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

def face_detection_func(event, context):

    body = json.loads(event['body'])
    content = base64.b64decode(body['content'])
    request_id = body['request_id']
    filename = body['filename']

    image = Image.open(BytesIO(content)).convert("RGB")
    image = np.array(image)
    image = Image.fromarray(image)

    # Face detection
    face, prob = mtcnn(image, return_prob=True, save_path=None)

    if face != None:

        face_img = face - face.min()  # Shift min value to 0
        face_img = face_img / face_img.max()  # Normalize to range [0,1]
        face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()  # Convert to uint8
        face_pil = Image.fromarray(face_img, mode="RGB")
        buffer = BytesIO()
        face_pil.save(buffer, format="PNG")
        encoded_face = base64.b64encode(buffer.getvalue()).decode("utf-8")

        sqs.send_message(
            QueueUrl=REQUEST_QUEUE_URL,
            MessageBody=json.dumps({
                'request_id': request_id,
                'face': encoded_face
            })
        )
    
    return {'statusCode': 200}