from io import BytesIO
import json
import boto3
import torch
import base64
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image

sqs = boto3.client(
    'sqs',
    region_name='us-east-1', 
    aws_access_key_id="",
    aws_secret_access_key=""
)
RESPONSE_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/430118847891/11111-res-queue'
resnet = InceptionResnetV1(pretrained='vggface2').eval()
saved_data = torch.load('resnetV1_video_weights.pt')  # loading resnetV1_video_weights.pt

def face_recognition_func(event, context):
    for record in event['Records']:
        message = json.loads(record['body'])
        request_id = message['request_id']
        face_b64 = message['face']

        # Load face image
        face_bytes = base64.b64decode(face_b64)
        face_pil = Image.open(BytesIO(face_bytes)).convert("RGB")

        # Step 2: Convert PIL to NumPy array (H, W, C) in range [0, 255]
        face_numpy = np.array(face_pil, dtype=np.float32)  # Convert to float for scaling

        # Step 3: Normalize values to [0,1] and transpose to (C, H, W)
        face_numpy /= 255.0  # Normalize to range [0,1]

        # Convert (H, W, C) → (C, H, W)
        face_numpy = np.transpose(face_numpy, (2, 0, 1))

        # Step 4: Convert NumPy to PyTorch tensor
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        if face_tensor != None:
            emb             = resnet(face_tensor.unsqueeze(0)).detach()  # detech is to make required gradient false
            embedding_list  = saved_data[0]  # getting embedding data
            name_list       = saved_data[1]  # getting list of names
            dist_list       = []  # list of matched distances, minimum distance is used to identify the person

            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))
        
            # Send result to response queue
            sqs.send_message(
                QueueUrl=RESPONSE_QUEUE_URL,
                MessageBody=json.dumps({
                    'request_id': request_id,
                    'result': name_list[idx_min]
                })
            )
    
    return {'statusCode': 200}