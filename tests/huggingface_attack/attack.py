import tomli
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import nanogcg_redteam
from nanogcg_redteam import GCGConfig

# Load configuration
with open("config.toml", "rb") as f:
    config_toml = tomli.load(f)

# 1. Setup Device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Running attack using device: {device}")

# 2. Load Model
model_id = config_toml["attack"]["model"]["model_id"]
print(f"Loading model: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
if device != "cuda":
    model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Configure GCG
# We rely on the standard configuration, overriding only what's in our config.toml
config = GCGConfig(
    verbosity="INFO",
    num_steps=config_toml["attack"]["parameters"]["num_steps"],
    search_width=config_toml["attack"]["parameters"]["search_width"],
    topk=config_toml["attack"]["parameters"]["topk"],
    batch_size=config_toml["attack"]["parameters"].get("batch_size"),
)

# 4. Run Attack
messages = [{"role": "user", "content": config_toml["attack"]["prompts"]["prompt"]}]
target_str = config_toml["attack"]["prompts"]["target"]

print("Starting GCG optimization (Standard HuggingFace Mode)...")
result = nanogcg_redteam.run(model, tokenizer, messages, target_str, config=config)

print("\n--- Optimization Complete ---")
print(f"Best String: {result.best_string}")
print(f"Best Loss: {result.best_loss}")

# 5. Final Verification (Local)
print("\n" + "=" * 50)
print("FINAL VERIFICATION (LOCAL MODEL)")
print("=" * 50)

# Construct the final prompt
final_prompt_messages = [{"role": "user", "content": config_toml["attack"]["prompts"]["prompt"].replace("{optim_str}", result.best_string)}]
final_prompt = tokenizer.apply_chat_template(final_prompt_messages, tokenize=False, add_generation_prompt=True)

print(f"\nPrompting Model with:\n{final_prompt}\n")

input_ids = tokenizer(final_prompt, return_tensors="pt").to(device)
output = model.generate(
    **input_ids,
    max_new_tokens=100,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)

print("-" * 20 + " MODEL GENERATION " + "-" * 20)
print(generated_text)
print("-" * 60)
print("=" * 50)
