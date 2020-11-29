#
# Copyright xiaomzho
#

import os
import greengrasssdk
from threading import Timer
from os.path import isfile, join, splitext
import time
import cv2
import numpy as np
from threading import Thread
import face_recognition
import urllib.request
import boto3
import scipy.misc
import threading
import sys
import random
import base64
import json
from botocore.exceptions import ClientError

from PIL import Image,ImageFont,ImageDraw
import schedule
import multiprocessing as mp

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

iotTopic = 'face_recognition'
errTopic = 'recognition_failed'
dist = 0

UPLOAD_TO_S3 = True

secret_name = "gg-lambda-access-s3-secretkey"
region_name = "ap-southeast-1"
secret = ''
access_key_id=''
access_secret_key=''

def get_secret():
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        print(e)
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            global secret
            secret = get_secret_value_response['SecretString']
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])

get_secret()
secret_json = json.loads(secret)
access_key_id = secret_json['access_key_id']
access_secret_key = secret_json['access_secret_key']

clientS3 = boto3.client(
    's3',
    aws_access_key_id = access_key_id,
    aws_secret_access_key = access_secret_key
)

bucket='greengrass-detect-realtime-video'
filesUrl = ['<RASPBERRYPI_LOCAL_FACE_IMAGE>']
faces_dict = {}

def frame_input(q):
    cap = cv2.VideoCapture('/dev/video0')
    #cap = cv2.VideoCapture('rtsp://localhost:8554/unicast')
    print("Raspberry Pi 4B - connected")
    
    while True:
        st = time.time()
        ret,frm = cap.read()
        if not(ret):
            cap.release()
            cap = cv2.VideoCapture('/dev/video0')
            #cap = cv2.VideoCapture("rtsp://localhost:8554/unicast")
            print("total time lost due to reinitialization : ",time.time()-st)
            continue
        q.put(frm)
        if q.qsize() > 1:
            for i in range(q.qsize()-1):
                q.get()

class Frame_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)
        
    def run(self):
        print("start queue read frame")
        mp.set_start_method('fork',True)
        process = mp.Process(target=frame_input,args=(queue,))
        process.daemon = True
        process.start()
        
queue = mp.Queue(maxsize=4)
frame_thread=Frame_Thread()
frame_thread.start()

print("started read read-time video frame from src [camera device / rtsp server]")
frame = queue.get()
ret,jpeg = cv2.imencode('.jpg', frame) 

def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def load_local_image(bucket,filesToLoad,newFile):
    global filesUrl
    global faces_dict
    for url in filesToLoad:
        img=scipy.misc.imread(url,mode='RGB')
        faces_dict.update({remove_file_ext(url):face_recognition.face_encodings(img)[0]})
        if(newFile):
            filesUrl.append(url)
        client.publish(topic=iotTopic, payload="images are loaded from local")

load_local_image(bucket,filesUrl,0)

def greengrass_infinite_infer_run():
    print("greengrass_infinite_infer_run() start...")
    try:
        global filesUrl
        #global images #schedule.every(300).seconds.do(loadS3Images)
        input_width=300
        input_height=300
        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Face detection starts now")
        prob_thresh = 0.25
        #results_thread = FIFO_Thread()
        #results_thread.start()
        #print("FIFO_THread inited") 
        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 0}
        client.publish(topic=iotTopic, payload="Model loaded")
        #ret, frame = cap.read()
        frame = queue.get()
        #if ret == False:
        if frame is None:
            print("queue.get() is None")
        if len(frame) == 0:
            raise Exception("Failed to get frame from the stream")
        doInfer = True
        frame_shape_msg = "frame shape is : " + str(frame.shape[0]) +" : "+ str(frame.shape[1])
        client.publish(topic=iotTopic, payload = frame_shape_msg)
        
        #doInfer = True
        #yscale = float(frame.shape[0]/input_height)
        #xscale = float(frame.shape[1]/input_width)
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        print("greengrass_infinite_infer_run while True...")
        while True:
            #schedule.run_pending()
            time.sleep(0.1)
            
            if(queue.qsize()==0):
                print("queue size == ",queue.qsize())
                continue
            
            frame = queue.get()
            if frame is None:
                client.publish(topic=errTopic, payload="Failed to get frame from the stream")
                continue
                #raise Exception("Failed to get frame from the stream")
            else:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                # Only process every other frame of video to save time
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    print('process this frame start:',len(face_locations),len(face_encodings))
                    face_names = []
                    for face_encoding in face_encodings:
                        name = "Unknown"
                        images_encodings = list(faces_dict.values())
                        global dist
                        dist = 0
                        match_result = face_recognition.compare_faces(images_encodings,face_encoding,tolerance=0.45)
                        for idx, match in enumerate(match_result):
                            if match:
                                image_encoding = images_encodings[idx]
                                dist = face_recognition.face_distance([image_encoding],face_encoding)[0]
                                dist = (1.0 - dist) * 100
                                print("name : {} face_recogniton dist value : {}".format(list(faces_dict.keys())[idx],dist))
                                if dist < 70.0:
                                    match_result[idx]=False
                                    name = ""
                                else :
                                    name = list(faces_dict.keys())[idx]
                        face_names.append(name)
                        msg = '{{"FaceName":"{0}","dist":"{1}","time":"{2}"}}'.format(str(name),str(dist),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
                        client.publish(topic=iotTopic, payload = msg)
                        #process_this_frame = not process_this_frame
                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                            cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
                            font_arial_path='/home/pi/Arial.ttf'
                            font_arial=ImageFont.truetype(font_arial_path,30)
                            img_pil= Image.fromarray(frame)
                            draw=ImageDraw.Draw(img_pil)
                            print("name=",name)
                            draw.text((left+10,bottom-32),name,fill=(255,255,255),font=font_arial)
                            frame=np.array(img_pil)
                            #font = cv2.FONT_HERSHEY_SIMPLEX
                            #cv2.putText(frame, name, (left + 10, bottom - 32), font, 3, (255, 255, 255), 2)

            global jpeg
            if dist > 70.0 :
                try:
                    imgID = time.strftime("%Y%m%d%H%M%S")+str(random.randint(0,99))
                    print("push image to S3 : ","dist=",dist,"imgID=",imgID)
                    s3Resp =clientS3.put_object(Bucket='greengrass-detect-realtime-video', Key=imgID+'.jpg', Body=jpeg.tobytes(), ACL="private")
                    print(s3Resp)
                    msg = '{{"FaceName":"{0}","dist":"{1}","imageName":"{2}","time":"{3}","desc":"{4}"}}'.format(str(name),str(dist),(imgID+".jpg"),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),"Uploaded Rapsberry Pi face detection image.")
                    client.publish(topic=iotTopic, payload=msg)
                except IOError as e:
                    print("Upload image failed:",str(e))
                    client.publish(topic=iotTopic, payload="Upload Rapsberry Pi face detection image failed.Exception:"+str(e))
                dist = 0 
            else:
                ret,jpeg = cv2.imencode('.jpg',frame)
            #ret,jpeg = cv2.imencode('.jpg', cv2.resize(frame, (0, 0), fx=1, fy=1))
    except Exception as e:
        msg = "greengrass_infinite_infer_run failed: " + str(e)
        client.publish(topic=errTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    #Timer(2, greengrass_infinite_infer_run).start()

# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return
