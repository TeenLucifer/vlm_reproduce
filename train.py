import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import PreTrainedModel, AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
from PIL import Image
from typing import List, Dict, Any
import logging

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, 
                 llm_model_path="./Qwen2.5-0.5B-Instruct",
                 vision_model_path="./siglip-base-patch16-224",
                 freeze_vision_model=True,
                 freeze_llm_model=True,
                 image_pad_num=49,
                 **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm_model = freeze_llm_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)

class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path) # image_size 224 patch_size 16 hidden_size 768
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path) # hidden_size 896
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if self.config.freeze_llm_model:
            for param in self.llm_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, labels, pixel_values, attention_mask=None): # image 224 * 224
        text_embeds = self.llm_model.get_input_embeddings()(input_ids) # (b, seq_len, 896)

        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 
        b, t, c = image_embeds.shape # b N t 196 c 768
        image_embeds = image_embeds.view(b, -1, 4*c) # (b, 196, 768) -> (b, 49, 3072)
        image_embeds = self.linear2(F.silu(self.linear1(image_embeds))) # (b, 49, 3072) -> (b, 49, 896)

        text_embeds = text_embeds.to(image_embeds.dtype)

        inputs_embeds = self.merge_text_embeds_with_image_embeds(text_embeds, image_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_function(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_text_embeds_with_image_embeds(self, text_embeds, image_embeds, input_ids):
        b, t, c = image_embeds.shape # (b, 49, 896)
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])

        text_embeds[batch_indices, image_indices] = image_embeds.view(-1, c)
        return text_embeds
    
class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.image_path = images_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        
        image_name = sample["image"]
        conversations = sample["conversations"]
        message = [
            {"role": "system", "content": "You are a helpful image recognization assistant."},
            {"role": "user", "content": conversations[0]["value"]}
        ]
        q_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
        q_input_ids = self.tokenizer(q_text)["input_ids"]
        a_text = conversations[1]["value"] + self.tokenizer.eos_token
        a_input_ids = self.tokenizer(a_text)["input_ids"]
        input_ids = q_input_ids + a_input_ids
        labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
        input_ids = input_ids[:-1]
        labels = labels[1:]
        
        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        pixel_values = self.processor(text=None, images=image)["pixel_values"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values
        }
    
class MyDataCollector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * (max_len - len(feature["input_ids"])))
            labels.append(feature["labels"] + [self.tokenizer.pad_token_id] * (max_len - len(feature["labels"])))
            pixel = feature["pixel_values"]
            if not isinstance(pixel, torch.Tensor):
                pixel = torch.tensor(pixel, dtype=torch.float32)
            if pixel.ndim == 3:
                pixel = pixel.unsqueeze(0)
            pixel_values.append(pixel)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.cat(pixel_values, dim=0)
        }
    
if __name__ == '__main__':
    config = VLMConfig()
    model = VLM(config=config).cuda()
    print(model)
    print(f"model parameters size: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")
    image_path = "./LLaVA-CC3M-Pretrain-595K/images"
    data_path = "./Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat_translated.json"
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = "save/pretrain"
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-6,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=16
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(image_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollector(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("save/pretrain")
    trainer.save_state()