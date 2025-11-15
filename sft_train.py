import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments
from typing import List, Dict, Any
from train import VLMConfig, VLM

def find_assistant_tokens(tokenizer, target):
    result = []
    start_index = 0
    end_index = 0
    while start_index <= len(target) - 1:
        if target[start_index]!=tokenizer("assistant")["input_ids"][0]:
            start_index+=1
            end_index+=1
        else:
            end_index+=1
            if target[end_index]==tokenizer("<|im_end|>")["input_ids"][0]:
                result.append((start_index+1, end_index+1))
                start_index=end_index+1
    return result

class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.images_path = images_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.datas = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        image_name = str(sample['image'])
        conversations = sample['conversations']
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]
        for conversation in conversations:
            if conversation['role'] == 'user':
                messages.append({'role': 'user', 'content': conversation['content']})
            else:
                messages.append({'role': 'assistant', 'content': conversation['content']})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
        input_ids = self.tokenizer(text)['input_ids']
        indexs = find_assistant_tokens(self.tokenizer, input_ids)
        labels = len(input_ids) * [self.tokenizer.pad_token_id]
        for index in indexs:
            labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
        input_ids = input_ids[:-1]
        labels = labels[1:]

        image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
        pixel_values = self.processor(text=None, images=image)['pixel_values']

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }
    
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel = feature["pixel_values"]
            if not isinstance(pixel, torch.Tensor):
                pixel = torch.tensor(pixel, dtype=torch.float32)
            if pixel.ndim == 3:
                pixel = pixel.unsqueeze(0)
            pixel_values.append(pixel)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }
    
if __name__ == '__main__':
    config = VLMConfig()
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    #model = AutoModelForCausalLM.from_config("./save/pretrain")
    model = AutoModelForCausalLM.from_pretrained("./save/pretrain")
    
    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model':
            param.requires_grad = False
        if 'llm_model' in name:
            param.requires_grad = True
    print(f"模型参数量为: {sum(p.numel() for p in model.parameters())}")
    print(f"模型可训练参数量为: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    images_path = './minimind-v_dataset/sft_images/sft_images'
    data_path = './minimind-v_dataset/sft_data.jsonl'
    output_dir = 'save/sft'
    args = TrainingArguments(
        output_dir = output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        learning_rate=1e-7,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SFTDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/sft')
    trainer.save_state()