import tomli  # type: ignore
import torch
from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

app = FastAPI()

# Load configuration
with open("config.toml", "rb") as f:
    config = tomli.load(f)

# Load a lightweight model for the mock API
MODEL_ID = config["server"]["model"]["model_id"]

# Automatically select the best available device:
# - "cuda" for NVIDIA GPUs
# - "mps" for Apple Silicon (M1/M2/M3/M4)
# - "cpu" as a fallback
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Loading mock API model {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
print("Model loaded.")


class PromptRequest(BaseModel):
    data: list[str]


@app.post("/predict")
async def predict(request: PromptRequest):
    try:
        prompt = request.data[0]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"data": [response_text]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run(app, host="0.0.0.0", port=8000)
