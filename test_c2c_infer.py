import torch
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model

checkpoint_dir = snapshot_download(
    repo_id="nics-efc/C2C_Fuser",
    allow_patterns=["qwen3_0.6b+qwen2.5_0.5b_Fuser/*"],
)

model_config = {
    "rosetta_config": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "checkpoints_dir": f"{checkpoint_dir}/qwen3_0.6b+qwen2.5_0.5b_Fuser/final",
    }
}

rosetta_model, tokenizer = load_rosetta_model(
    model_config,
    eval_config={},
    device=torch.device("cuda")
)

device = rosetta_model.device
prompt = [{"role": "user", "content": "Say hello in one short sentence."}]
input_text = tokenizer.apply_chat_template(
    prompt,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
inputs = tokenizer(input_text, return_tensors="pt").to(device)

instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(
    inputs["input_ids"].shape[1] - 1, 1
).unsqueeze(0).to(device)
label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
kv_cache_index = [instruction_index, label_index]

with torch.no_grad():
    outputs = rosetta_model.generate(
        **inputs,
        kv_cache_index=kv_cache_index,
        do_sample=False,
        max_new_tokens=64,
    )
    output_text = tokenizer.decode(
        outputs[0, instruction_index.shape[1] + 1:],
        skip_special_tokens=True
    )
    print(output_text)
