import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation, \
    AutoModelForTableQuestionAnswering
import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import os
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import uuid

from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import cv2
import einops
from pytorch_lightning import seed_everything
import random
from ldm.util import instantiate_from_config
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
from ControlNet.annotator.util import HWC3, resize_image
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.annotator.uniformer import UniformerDetector
import whisper
from TTS.api import TTS
import pandas as pd

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    return os.path.join(head, new_file_name)


def create_model(config_path, device):
    config = OmegaConf.load(config_path)
    OmegaConf.update(config, "model.params.cond_stage_config.params.device", device)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


class MaskFormer:
    def __init__(self, device):
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt", ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(image.size)


class Pix2Pix:
    def __init__(self, device):
        print("Initializing Pix2Pix to %s" % device)
        self.device = device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           torch_dtype=torch.float16,
                                                                           safety_checker=None).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting Pix2Pix Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = \
            self.pipe(instruct_text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2, ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        return updated_image_path


class T2I:
    def __init__(self, device):
        print("Initializing T2I to %s" % device)
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model,
                                              tokenizer=self.text_refine_tokenizer, device=self.device)
        self.pipe.to(device)

    def inference(self, text):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        refined_text = self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        print(f'{text} refined to {refined_text}')
        image = self.pipe(refined_text).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename


class ImageCaptioning:
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
            self.device)

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions


class image2pose:
    def __init__(self):
        print("Direct human pose.")
        self.detector = OpenposeDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2pose Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path


class pose2image:
    def __init__(self, device):
        print("Initialize the pose2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_openpose.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, ' \
                        'cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting pose2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [
            self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control],
                   "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in
                                     range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False,
                                                          eta=0., unconditional_guidance_scale=self.scale,
                                                          unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path


class image2seg:
    def __init__(self):
        print("Direct segmentations.")
        self.detector = UniformerDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2seg Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path


class seg2image:
    def __init__(self, device):
        print("Initialize the seg2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_seg.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, ' \
                        'cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting seg2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [
            self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control],
                   "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in
                                     range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False,
                                                          eta=0., unconditional_guidance_scale=self.scale,
                                                          unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path


class BLIPVQA:
    def __init__(self, device):
        print("Initializing BLIP VQA to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)

    def get_answer_from_question_and_image(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert('RGB')
        print(F'BLIPVQA :question :{question}')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer


class Whisper:
    def __init__(self, device):
        print("Initializing Whisper on device", device)
        self.model = whisper.load_model("medium.en", device=device)

    def transcribe(self, inputs):
        return self.model.transcribe(inputs)['text']

class coqui_tts:

    def __init__(self, device):
        self.device = device
        self.tts = TTS('tts_models/multilingual/multi-dataset/your_tts', gpu=self.device)

    def gen_speech_from_text(self, inputs):
        print("===>Starting text2speech Inference")
        filename = os.path.join('audio', str(uuid.uuid4())[:8] + ".wav")
        self.tts.tts_to_file(text=inputs, speaker=self.tts.speakers[0], language=self.tts.languages[0],
                             file_path=filename)

        return filename


class TableQA:

    def __init__(self, device):
        self.device = device
        self.pipeline = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq", device=self.device)

    def get_answer_from_question_and_table(self, inputs):
        table_path = inputs.split(",")[0]
        questions = inputs.split(",")[1:]
        table = pd.read_csv(table_path, dtype=str)

        res = self.pipeline(table=table, query=questions)

        return res['answer']


class TwilioCaller:
    def parse_input(self, inputs):
        try:
            if 'and' in inputs:
                text: str = inputs.split("and")[0]
                phone_number = inputs.split("and")[1]
            elif ',' in inputs:
                text: str = inputs.split(",")[0]
                phone_number = inputs.split(",")[1]
            else:
                raise Exception('Could not make the call, the input is not well formatted. Must be a comma separated string')
        except:
            raise Exception('Could not parse your input. Must be a comma separated string')
        text = text.replace('"', '').strip(' ')
        phone_number = phone_number.replace('"', '').strip(' ')
        if not re.match('\+[0-9]+', text) and not re.match('\+[0-9]+', phone_number):
            raise Exception('Could not make the call, no phone number provided')
        if re.match('\+[0-9]+', text) and not re.match('\+[0-9]+', phone_number):
            text, phone_number = phone_number, text
        return text, phone_number

    def call_with_text(self, inputs):
        import twilio
        try:
            text, phone_number = self.parse_input(inputs)
        except Exception as e:
            return str(e)
        from twilio_lib import call_with_text
        try:
            call_with_text(text, phone_number)
        except twilio.base.exceptions.TwilioRestException:
            return 'Internal error, could not submit the call.'

        return 'Call submitted, it should be received soon'

    def call_with_audio(self, inputs):
        audio_filename = inputs.split(",")[0]
        phone_number = inputs.split(",")[1:]
        from twilio_lib import call_with_audio
        call_with_audio(audio_filename, phone_number)

        return 'Call submitted, it should be received soon'


class ConversationBot:
    def __init__(self):
        print("Initializing VisualChatGPT")
        self.llm = OpenAI(temperature=0)
        # self.i2t = ImageCaptioning(device="cuda:1")  # 1755
        # self.t2i = T2I(device="cuda:1")  # 6677
        # self.image2pose = image2pose()
        # self.pose2image = pose2image(device="cuda:1")  # 6681
        # self.BLIPVQA = BLIPVQA(device="cuda:1")  # 2709
        # self.image2seg = image2seg()
        # self.seg2image = seg2image(device="cuda:1")  # 5540
        # ## up until now, comsuming  23362 MB on GPU
        # self.pix2pix = Pix2Pix(device="cuda:2")  # 2795
        # self.coqui_tts = coqui_tts(device=False)
        # self.tableQA = TableQA(device="cuda:2")
        # self.whisper = Whisper(device="cuda:2")
        # self.twilio_caller = TwilioCaller()

        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            # Tool(name="Get Photo Description", func=self.i2t.inference,
            #      description="useful when you want to know what is inside the photo. receives image_path as input. "
            #                  "The input to this tool should be a string, representing the image_path. "),
            # Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
            #      description="useful when you want to generate an image from a user input text and save it to a file. "
            #                  "like: generate an image of an object or something, or generate an image that includes "
            #                  "some objects."
            #                  "The input to this tool should be a string, representing the text used to generate image. "),
            #
            # Tool(name="Instruct Image Using Text", func=self.pix2pix.inference,
            #      description="useful when you want to the style of the image to be like the text. like: make it look "
            #                  "like a painting. or make it like a robot."
            #                  "The input to this tool should be a comma separated string of two, representing the "
            #                  "image_path and the text. "),
            # Tool(name="Answer Question About The Image", func=self.BLIPVQA.get_answer_from_question_and_image,
            #      description="useful when you need an answer for a question based on an image. like: what is the "
            #                  "background color of the last image, how many cats in this figure, what is in this figure."
            #                  "The input to this tool should be a comma separated string of two, representing the "
            #                  "image_path and the question"),
            # Tool(name="Segmentation On Image", func=self.image2seg.inference,
            #      description="useful when you want to detect segmentations of the image. like: segment this image, "
            #                  "or generate segmentations on this image, or perform segmentation on this image."
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Segmentations", func=self.seg2image.inference,
            #      description="useful when you want to generate a new real image from both the user description and "
            #                  "segmentations. like: generate a real image of a object or something from this "
            #                  "segmentation image, or generate a new real image of a object or something from these "
            #                  "segmentations."
            #                  "The input to this tool should be a comma separated string of two, representing the "
            #                  "image_path and the user description"),
            # Tool(name="Pose Detection On Image", func=self.image2pose.inference,
            #      description="useful when you want to detect the human pose of the image. like: generate human poses "
            #                  "of this image, or generate a pose image from this image."
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Pose Image", func=self.pose2image.inference,
            #      description="useful when you want to generate a new real image from both the user description and a "
            #                  "human pose image. like: generate a real image of a human from this human pose image, "
            #                  "or generate a new real image of a human from this pose."
            #                  "The input to this tool should be a comma separated string of two, representing the "
            #                  "image_path and the user description"),
            # Tool(name="Generate Text from Audio", func=self.whisper.transcribe,
            #      description="useful when you want to generate text from audio. like: generate text from this audio, or transcribe this audio, or listen to this audio. receives audio_path as input."
            #                  "The input to this tool should be a string, representing the audio_path"),
            # Tool(name="Generate Text From Speech", func=self.coqui_tts.gen_speech_from_text,
            #      description="useful when you want to generate a speech from a text. like: generate a speech from "
            #                  "this text, or generate a speech from this sentence. "
            #                  "The input to this tool should be a string, representing the text to be converted to "
            #                  "speech."
            #      ),
            # Tool(name="Answer Question About The table", func=self.tableQA.get_answer_from_question_and_table,
            #      description="useful when you need an answer for a question based on a table. like: what is the "
            #                  "maximum of the column age, or  what is the sum of row 5 from the following table."
            #                  "The input to this tool should be a comma separated string, representing the "
            #                  "table_path and the questions"),
            # Tool(name="Call a phone number with text", func=self.twilio_caller.call_with_text,
            #      description="useful when you need to call a phone number with a text input. like: call +4917686490193 and"
            #                  " tell him \"happy birthday\". The input to this tool should be a comma separate string "
            #                  "representing the text_input and the phone_number"),
            # Tool(name="Call a phone number with audio", func=self.twilio_caller.call_with_audio,
            #      description="useful when you need to call a phone number with an audio file. like: call +4917686490193 and"
            #                  " using audio file audio/smth.wav. Only use audio files mentioned by the user."
            #                  "The input to this tool should be a comma separated string representing the audio file name and the phone_number"),
        ]

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def run_text(self, text, state, audio):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        audio_files = re.findall('(audio/\S*wav)', response)
        if len(audio_files) > 0:
            audio = audio_files[0]
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state, audio

    def run_image(self, image, state, txt):
        print("===============Running run_image =============")
        print("Inputs:", image, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.i2t.inference(image_filename)
        Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(
            image_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + ' ' + image_filename + ' '

    def run_audio(self, audio, state, txt):
        print("===============Running run_audio =============")
        print("Inputs:", audio, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        import shutil
        shutil.copyfile(audio, audio_filename)
        transcribed_text = self.whisper.transcribe(audio_filename)
        Human_prompt = "\nHuman: provide audio named {}. The description is: {}. This information helps you to understand this audio, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(
            audio_filename, transcribed_text)

        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={audio_filename})*{audio_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, audio, state, txt + ' ' + audio_filename + ' '

    # bot.run_df, [persist_df, state, txt], [chatbot, persist_df, state, txt])

    def run_df(self, df, state, txt):
        print("===============Running run_df =============")
        print("Inputs:", df, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        csv_filename = os.path.join('csv', str(uuid.uuid4())[0:8] + ".csv")
        df.to_csv(csv_filename, index=False)
        Human_prompt = "\nHuman: provided a csv file named {}. You can specifically use the tool \"Answer Question About The table\" to understand this file. If you understand, say \"Received\". \n".format(
            csv_filename)

        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={csv_filename})*{csv_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + ' ' + csv_filename + ' '



if __name__ == '__main__':
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(placeholder="Enter text and press enter", label='Instruct with text').style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload Image", file_types=["image"])
        with gr.Row():
            with gr.Column(scale=0.25, min_width=0):
                input_audio = gr.Audio(source="microphone", label='Instruct with Audio')
            with gr.Column(scale=0.25, min_width=0):
                audio = gr.Audio(type="filepath", label='Audio output', interactive=False)
            with gr.Column(scale=0.5, min_width=0):
                with gr.Row():
                    df = gr.DataFrame(interactive=True, row_count=1, col_count=1, headers=['Column1'], label="Give a Dataframe as input")
                with gr.Row():
                    with gr.Column(scale=0.8, min_width=0):
                        persist_df = gr.Button("Upload the dataframe")

        audio.upload(bot.run_audio, [audio, state, txt], [chatbot, audio, state, txt])
        txt.submit(bot.run_text, [txt, state, audio], [chatbot, state, audio])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        persist_df.click(bot.run_df, [df, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)

        demo.launch(server_name="0.0.0.0", server_port=7860)
