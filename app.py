import gradio as gr

import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig

from train import VLMConfig, VLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("./siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained('./Qwen2.5-0.5B-Instruct')
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

max_new_tokens = 100
temperature = 0.0
eos = tokenizer.eos_token_id
top_k = None

def vqa_inference(image: Image.Image, query: str, model_choice: str = "sft") -> str:
    if image is None:
        return "请上传图片后再进行提问，需要图片信息才能回答问题。"

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(f"./save/{model_choice}")
    model.to(device)
    model.eval()

    # 图片转tensor
    pixel_values = processor(text=None, images=image).pixel_values
    pixel_values = torch.tensor(pixel_values).to(device)

    # 加载对话模板
    q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":f'{query}\n<image>'}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*49)
    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(device)

    s = input_ids.shape[1]
    while input_ids.shape[1] < s + max_new_tokens - 1:  
        inference_res = model(input_ids, None, pixel_values)  
        logits = inference_res.logits 
        logits = logits[:, -1, :] 

        for token in set(input_ids.tolist()[0]):  
            logits[:, token] /= 1.0

        if temperature == 0.0: 
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature  
            if top_k is not None:  
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') 

            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

        if idx_next == eos:  
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)  
    return tokenizer.decode(input_ids[:, s:][0])

# ===== 伪推理函数（按需替换为你的真实后端） =====
def run_inference(image: Image.Image | None, model_name: str, user_input: str):
    """
    这里用一个简单的示例逻辑：
    - 回显所选模型
    - 返回一个示例回答
    - 如果需要，也可以在这里对 image 做处理
    """
    if image is None and not user_input.strip():
        return "请先上传图片或填写输入文本。"

    reply = vqa_inference(image=image, query=user_input)
    return reply

# ===== 自定义样式 =====
CUSTOM_CSS = """
/* 整体留白 */
.gradio-container { max-width: 1200px; margin: 0 auto; }

/* 左侧上传卡片的占位样式，模拟“将图像拖放到此处/点击上传” */
.upload-card .wrap .container .upload-zone,
.upload-card .wrap {
    height: 360px;
}
.upload-card .wrap .container {
    height: 100%;
}
.upload-card .wrap .container .upload-zone {
    border: 1.5px dashed #d9d9d9 !important;
    display: flex; align-items: center; justify-content: center;
    color: #999;
    font-size: 14px;
}

/* 右侧表单区的卡片感 */
.right-card {
    border: 1px solid #eee;
    border-radius: 6px;
    padding: 16px;
    background: #fff;
}

/* “生成”按钮宽度与截图风格相近 */
.btn-full .gr-button { width: 100%; }
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("### 模型测试页面")

    with gr.Row():
        # ---------- 左列：图片上传 ----------
        with gr.Column(scale=1):
            gr.Markdown("**选择图片**")
            with gr.Row():
                image_in = gr.Image(
                    label=None,
                    sources=["upload", "clipboard", "webcam"],
                    type="pil",
                    height=360,
                    elem_classes=["upload-card"],  # for dashed styling
                )

        # ---------- 右列：模型与问答 ----------
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["right-card"]):
                model_choice = gr.Radio(
                    choices=["pretrain", "sft"],
                    value="sft",
                    label="选择模型",
                    interactive=True,
                )
                text_in = gr.Textbox(label="输入文本", lines=3, placeholder="在此输入你的问题…")
                text_out = gr.Textbox(label="输出文本", lines=6)

                with gr.Row(elem_classes=["btn-full"]):
                    run_btn = gr.Button("生成", variant="primary")

    # 交互：点击“生成”-> 执行推理并把结果写到输出框
    run_btn.click(
        fn=run_inference,
        inputs=[image_in, model_choice, text_in],
        outputs=[text_out],
        show_progress=True,
    )

# 入口
if __name__ == "__main__":
    demo.launch()