import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Say

def write_text_twilml(text):
    response = VoiceResponse()
    response.say(text)
    import tempfile

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        # Write a string to the file
        temp_file.write(str(response))

        # Get the file name
        file_name = temp_file.name
    return file_name

def write_voice_twilml(audio_url):
    response = VoiceResponse()
    response.play(audio_url)
    import tempfile

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        # Write a string to the file
        temp_file.write(str(response))

        # Get the file name
        file_name = temp_file.name
    return file_name

def push_to_s3(file_name, extension, content_type=None):
    import boto3
    import uuid

    s3 = boto3.client('s3')

    bucket_name = 'god-llm'
    file_name = file_name
    object_key = f'twilml/{str(uuid.uuid4())[0:4]}.{extension}'
    extra_args = {'ACL': 'public-read'}
    if content_type:
        extra_args["ContentType"] = content_type
    s3.upload_file(file_name, bucket_name, object_key, ExtraArgs=extra_args)
    return f'https://god-llm.s3.eu-central-1.amazonaws.com/{object_key}'

def call_with_twilml_url(twilml_url, phone_number):
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        method='GET',
        url=twilml_url,
        # to='+4917686490193',
        to=phone_number,
        from_='+15673393771'
    )
    return call

def call_with_text(text, phone_number):
    file_name = write_text_twilml(text)
    twilml_url = push_to_s3(file_name, extension='xml', content_type="text/xml")
    call_with_twilml_url(twilml_url, phone_number)


def call_with_audio(audio_file, phone_number):
    audio_url = push_to_s3(audio_file, extension='wav', content_type="audio/wav")
    file_name = write_voice_twilml(audio_url)
    twilml_url = push_to_s3(file_name, extension='xml', content_type="text/xml")
    call_with_twilml_url(twilml_url, phone_number)
