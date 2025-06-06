import json
import base64
import boto3
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN
import numpy as np
import time
from awscrt import mqtt
from awsiot import mqtt_connection_builder

sqs = boto3.client(
    'sqs',
    region_name='us-east-1', 
    aws_access_key_id="",
    aws_secret_access_key=""
)
# REQUEST_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/156081010342/1233725170-req-queue'
# RESPONSE_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/156081010342/1233725170-resp-queue'
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint="a1l5d26d1dd5qw-ats.iot.us-east-1.amazonaws.com",
    port=8883,
    cert_filepath="/greengrass/v2/thingCert.crt",
    pri_key_filepath="/greengrass/v2/privKey.key",
    root_ca_filepath="/greengrass/v2/rootCA.pem",
    client_id="1233725170-IoTThing",
    clean_session=False,
    keep_alive_secs=30
)

def on_message(topic, payload, **kwargs):
    body = json.loads(payload)
    face_detection_func(body)

connect_future = mqtt_connection.connect()
connect_future.result() 

subscribe_info = mqtt_connection.subscribe(
    topic="clients/1233725170-IoTThing",
    qos=mqtt.QoS.AT_LEAST_ONCE,
    callback=on_message
)
print(f"Subscribed with {subscribe_info}")

def face_detection_func(body):

    content = base64.b64decode(body['encoded'])
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
    else:
        sqs.send_message(
            QueueUrl=RESPONSE_QUEUE_URL,
            MessageBody=json.dumps({
                'request_id': request_id,
                'result': "No-Face"
            })
        )
    

while True:
    time.sleep(1)