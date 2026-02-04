import os
import sys

import torch
from modeling_urag import URaG_ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer


def chat_with_URaG(prompt, image_paths):
    im_cont = [{"type": "image", "image": imp} for imp in image_paths]
    messages = [
        {
            "role": "user",
            "content": im_cont + [{"type": "text", "text": prompt}]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, output_attentions=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return output_text[0]



if __name__ == '__main__':
    model_root = "URaG-3B"
    model = URaG_ForConditionalGeneration.from_pretrained(
        model_root, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_root, min_pixels=128*28*28, max_pixels=1024*28*28)
    
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    VQA_prompt = "\nPlease try to answer the question with short words or phrases if possible."
    
    
    model.remain_pages = 5
    model.layer_for_retrieval = 6
    model.end_question_position = -4 - len(tokenizer(VQA_prompt)['input_ids'])
    
    
    question = "How much is the Trading Operating Profit in 2011?"
    im_root = 'dataset/demo_ims'
    imps = [os.path.join(im_root, f'slide_{i}_1024.jpg') for i in range(1, 21)]
    
    output_text = chat_with_URaG(question + VQA_prompt, imps)
    print('output_text:', output_text)
    