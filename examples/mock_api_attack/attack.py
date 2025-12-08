import tomli  # type: ignore
import torch
from transformers import AutoTokenizer  # type: ignore
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

import nanogcg_redteam
from nanogcg_redteam import APITarget, GCGConfig, ProbeSamplingConfig

# Load configuration
with open("config.toml", "rb") as f:
    config_toml = tomli.load(f)

# 1. Setup Device

# Automatically select the best available device:
# - "cuda" for NVIDIA GPUs
# - "mps" for Apple Silicon (M1/M2/M3/M4)
# - "cpu" as a fallback
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Running attack using device: {device}")

# 2. Configure API Target (Mock Server)
print("Configuring API Target...")
api_target = APITarget(
    endpoint="http://localhost:8000/predict",
    payload_template={"data": ["{prompt}"]},
    prompt_placeholder="{prompt}",
    response_parser=lambda x: x["data"][0],
)

# 3. Configure Probe Sampling (Draft Model)
draft_model_id = config_toml["attack"]["models"]["draft_model_id"]
print(f"Loading draft model ({draft_model_id})...")
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_id, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
).to(device)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)

probe_sampling_config = ProbeSamplingConfig(
    draft_model=draft_model,
    draft_tokenizer=draft_tokenizer,
    r=config_toml["attack"]["parameters"]["r"],
    sampling_factor=config_toml["attack"]["parameters"]["sampling_factor"],
)

# 4. Configure GCG
config = GCGConfig(
    target=api_target,
    probe_sampling_config=probe_sampling_config,
    verbosity="INFO",
    num_steps=config_toml["attack"]["parameters"]["num_steps"],
    search_width=config_toml["attack"]["parameters"]["search_width"],
    topk=config_toml["attack"]["parameters"]["topk"],
)

# 5. Load Local Proxy Model (Mistral)
model_id = config_toml["attack"]["models"]["proxy_model_id"]
print(f"Loading proxy model: {model_id}...")
# Note: Mistral-7B is large. Ensure you have enough memory.
# Using float16 for GPU/MPS, float32 for CPU (though CPU will be very slow)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16 if device != "cpu" else torch.float32
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 6. Run Attack
messages = [{"role": "user", "content": config_toml["attack"]["prompts"]["prompt"]}]
target_str = config_toml["attack"]["prompts"]["target"]

print("Starting GCG optimization with Probe Sampling...")
result = nanogcg_redteam.run(model, tokenizer, messages, target_str, config=config)

print("\n--- Optimization Complete ---")
print(f"Best String: {result.best_string}")
print(f"Best Loss: {result.best_loss}")
print(
    f"Target Responses (Last 5): {result.target_results[-5:] if result.target_results else 'None'}"
)

# 7. Final Verification
print("\n" + "=" * 50)
print("FINAL VERIFICATION AGAINST API")
print("=" * 50)

# Construct the final prompt
final_prompt = messages[-1]["content"].replace("{optim_str}", result.best_string)
print(f"\nSending Final Prompt:\n{final_prompt}\n")

# Evaluate
print("-" * 20 + " API RESPONSE " + "-" * 20)
final_response = api_target.evaluate(final_prompt)
print(f"\n{final_response}\n")
print("=" * 50)
