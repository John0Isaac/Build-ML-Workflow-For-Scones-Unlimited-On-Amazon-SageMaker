"""
Serialize Image Data
"""

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    boto3.resource('s3').Bucket(bucket).download_file(key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        "image_data": image_data,
        "s3_bucket": bucket,
        "s3_key": key,
        "inferences": []
    }


"""
Image Classification
"""

import json
import base64
import boto3

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-06-16-18-57-58-104"

sagemaker = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"])
    
    # Make a prediction:
    inferences = sagemaker.invoke_endpoint(EndpointName=ENDPOINT, Body=image, ContentType='image/png')
    
    # We return the data back to the Step Function    
    event["inferences"] = json.loads(inferences['Body'].read().decode('utf-8'))
    return event

"""
Filter Low Confidence Inferences
"""

import json


THRESHOLD = .88


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise ValueError("THRESHOLD_CONFIDENCE_NOT_MET")

    return event