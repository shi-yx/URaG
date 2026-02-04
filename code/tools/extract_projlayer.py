import os
import argparse
import torch
from peft import PeftModel
import sys
sys.path.append('../')
from URaG_code import URaG_ForConditionalGeneration



def extract_projlayer(model_path, model_base, save_path):
    print('Loading Qwen2.5-VL from base model...')
    model = URaG_ForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0")
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional Qwen2.5-VL weights...')
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')

    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:].replace('.base_layer', '') if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    
    proj_layer = model.proj_layer
    torch.save(proj_layer.state_dict(), save_path)

    print(f'Proj_layer params saved to {save_path}!')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_base', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    
    extract_projlayer(args.model_path, args.model_base, args.save_path)