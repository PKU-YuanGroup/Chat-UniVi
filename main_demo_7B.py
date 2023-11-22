import torch
import gradio as gr
from fastapi import FastAPI
from ChatUniVi.conversation import conv_templates, Conversation
from ChatUniVi.demo import Chat
from ChatUniVi.constants import *
import os
from PIL import Image
import tempfile
import imageio
from decord import VideoReader, cpu


app = FastAPI()
model_path = "Chat-UniVi/Chat-UniVi"  # model_path = [model path]
assert model_path is not ""

def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0
    f_end = int(min(1000000000, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = 1
        t_stride = int(round(float(fps) / sample_fps))
        all_pos = list(range(f_start, f_end + 1, t_stride))
        sample_pos = all_pos
        patch_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]

    writer = imageio.get_writer(filename, format='FFMPEG', fps=8)
    for frame in patch_images:
        writer.append_data(frame)
    writer.close()
    return filename


def generate(image1, image2, video, textbox_in, first_run, state, state_, images_tensor):

    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    image1 = image1 if image1 else "none"
    image2 = image2 if image2 else "none"
    video = video if video else "none"

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = []

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "image")

    image_processor = handler.image_processor
    if os.path.exists(image1):
        images = [Image.open(image1)]
        images_tensor.append(image_processor(images, return_tensors='pt')['pixel_values'][0].to(handler.model.device, dtype=torch.float16))

    if os.path.exists(image2):
        images = [Image.open(image2)]
        images_tensor.append(image_processor(images, return_tensors='pt')['pixel_values'][0].to(handler.model.device, dtype=torch.float16))

    if os.path.exists(video):
        video_tensor = handler._get_rawvideo_dec(video, image_processor, max_frames=MAX_IMAGE_LENGTH)
        for img in video_tensor:
            images_tensor.append(image_processor(img, return_tensors='pt')['pixel_values'][0].to(handler.model.device, dtype=torch.float16))

    if os.path.exists(video):
        text_en_in = DEFAULT_IMAGE_TOKEN * len(video_tensor) + '\n' + text_en_in
    if os.path.exists(image2):
        text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in
    if os.path.exists(image1):
        text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in

    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    show_images = ""
    if os.path.exists(image1):
        filename = save_image_to_local(image1)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if os.path.exists(image2):
        filename = save_image_to_local(image2)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if os.path.exists(video):
        filename = save_video_to_local(video)
        show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True))


def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True), \
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        True, state, state_, state.to_gradio_chatbot(), [])


conv_mode = "simple"
handler = Chat(model_path, conv_mode=conv_mode)
if not os.path.exists("temp"):
    os.makedirs("temp")

with gr.Blocks(gr.themes.Soft()) as demo:
    demo.title = 'Demo'
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            image1 = gr.Image(label="Input Image1", type="filepath")
            image2 = gr.Image(label="Input Image2", type="filepath")
            video = gr.Video(label="Input Video")

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="Chat-UniVi", bubble_full_width=True).style(height=1200)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox = gr.Textbox(label="Input Text")
                with gr.Column(scale=1, min_width=60, label="Input Text"):
                    submit_btn = gr.Button(value="Submit", visible=True)
            with gr.Row(visible=True) as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    submit_btn.click(generate, [image1, image2, video, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, image2, video])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        generate, [image1, image2, video, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, image2, video])

    clear_btn.click(clear_history, [state, state_],
                    [image1, image2, video, textbox, first_run, state, state_, chatbot, images_tensor])

app = gr.mount_gradio_app(app, demo, path="/")