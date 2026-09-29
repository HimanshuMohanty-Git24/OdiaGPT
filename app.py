"""
OdiaGPT Interactive Gradio Web UI
Built from Scratch - 100% PyTorch Primitives & From-Scratch Indic Tokenizer
"""

from pathlib import Path
import time
import json
import sys

import gradio as gr
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from odiagpt.checkpoint import load_latest_checkpoint
from odiagpt.config import ModelConfig
from odiagpt.model import OdiaGPT

# Global models and tokenizer cache
MODELS = {}
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        tok_info_path = ROOT / "tokenizer/nano_8k.json"
        tok_info = json.loads(tok_info_path.read_text(encoding="utf-8"))
        TOKENIZER = spm.SentencePieceProcessor(model_file=tok_info["model_path"])
    return TOKENIZER


def get_model(model_name: str):
    global MODELS
    if model_name in MODELS:
        return MODELS[model_name]

    if "30M" in model_name:
        exp_id = "pilot-30m-v1"
    else:
        exp_id = "nano-8k-v1"

    checkpoint = load_latest_checkpoint(ROOT, exp_id, map_location="cpu")
    if checkpoint is None:
        raise RuntimeError(f"Checkpoint for {exp_id} not found")
    saved, _ = checkpoint
    config = ModelConfig.from_dict(saved["model_config"])
    model = OdiaGPT(config).to(DEVICE).eval()
    model.load_state_dict(saved["model"])
    MODELS[model_name] = (model, config, saved["step"])
    return MODELS[model_name]


@torch.inference_mode()
def generate_text(
    prompt: str,
    model_choice: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
):
    if not prompt or not prompt.strip():
        return "ଦୟାକରି ଏକ Odia prompt ଲେଖନ୍ତୁ (Please provide an Odia prompt).", "0 tokens", "0.0s", "0.0 tok/s"

    sp = get_tokenizer()
    model, config, step = get_model(model_choice)

    prompt_ids = [sp.bos_id(), *sp.encode(prompt, out_type=int)]
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    if seed == -1:
        seed = torch.randint(0, 100000, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    start_time = time.monotonic()
    generated_count = 0

    for _ in range(max_new_tokens):
        context = ids[:, -config.context_length:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :].float().clone()
        
        # Suppress special tokens
        next_logits[:, [sp.pad_id(), sp.unk_id(), sp.bos_id()]] = -torch.inf

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(ids[0].tolist()):
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        if temperature == 0.0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits /= max(temperature, 1e-4)
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                boundary = torch.topk(next_logits, k=k, dim=-1).values[:, -1:]
                next_logits = next_logits.masked_fill(next_logits < boundary, -torch.inf)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
                next_logits = torch.full_like(next_logits, -torch.inf).scatter(
                    dim=-1, index=sorted_indices, src=sorted_logits
                )
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)

        ids = torch.cat((ids, next_id), dim=1)
        generated_count += 1
        if int(next_id.item()) == sp.eos_id():
            break

    elapsed = time.monotonic() - start_time
    tok_per_sec = generated_count / max(elapsed, 1e-5)

    new_ids = ids.squeeze(0).tolist()[len(prompt_ids):]
    decode_ids = [i for i in new_ids if i != sp.eos_id()]
    completion = sp.decode(decode_ids)
    full_text = prompt + completion

    stats_tokens = f"{generated_count} tokens"
    stats_time = f"{elapsed:.2f}s"
    stats_speed = f"{tok_per_sec:.1f} tok/s"

    return full_text, stats_tokens, stats_time, stats_speed


title_html = """
<div style="text-align: center; padding: 18px; background: linear-gradient(135deg, #3B0764 0%, #1E1B4B 100%); border-radius: 12px; color: white; margin-bottom: 20px;">
    <h1 style="font-size: 2.2rem; margin: 0; font-weight: 700; color: #FAF5FF;">ଓଡ଼ିଆଜିପିଟି (OdiaGPT)</h1>
    <p style="font-size: 1.05rem; margin-top: 6px; color: #E9D5FF;">
        First Odia Decoder-Only Language Model Built Completely From Scratch
    </p>
    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
        <span style="background: rgba(255, 255, 255, 0.15); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">⚡ 31.5M Parameters</span>
        <span style="background: rgba(255, 255, 255, 0.15); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🧠 RoPE + RMSNorm + SwiGLU</span>
        <span style="background: rgba(255, 255, 255, 0.15); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">📖 Scratch 8K SentencePiece</span>
        <span style="background: rgba(255, 255, 255, 0.15); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🚀 100% PyTorch Primitives</span>
    </div>
</div>
"""

with gr.Blocks(title="OdiaGPT — Scratch Odia Transformer") as demo:
    gr.HTML(title_html)

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="ପ୍ରବେଶ କରନ୍ତୁ (Enter Odia Prompt)",
                placeholder="ଯେପରିକି: ଓଡ଼ିଶାର ଲୋକମାନେ / ଆଜି ସକାଳେ / ଜଗନ୍ନାଥ ମନ୍ଦିର...",
                lines=3,
                value="ଓଡ଼ିଶାର ଲୋକମାନେ",
            )
            generate_btn = gr.Button("ଉତ୍ପାଦନ କରନ୍ତୁ (Generate Completion) 🚀", variant="primary", size="lg")

            with gr.Accordion("ଉଦାହରଣ (Example Prompts)", open=True):
                gr.Examples(
                    examples=[
                        ["ଓଡ଼ିଶାର ଲୋକମାନେ"],
                        ["ଆଜି ସକାଳେ"],
                        ["ଓଡ଼ିଆ ଭାଷାର"],
                        ["ଜଗନ୍ନାଥ ମନ୍ଦିର ଏକ"],
                        ["ଭୁବନେଶ୍ୱର ସହରରେ"],
                    ],
                    inputs=[prompt_input],
                )

        with gr.Column(scale=5):
            output_text = gr.Textbox(
                label="ଫଳାଫଳ (Generated Output)",
                lines=7,
            )
            with gr.Row():
                tokens_out = gr.Textbox(label="Tokens Generated", interactive=False)
                time_out = gr.Textbox(label="Latency", interactive=False)
                speed_out = gr.Textbox(label="Speed", interactive=False)

    with gr.Accordion("⚙️ ନିୟନ୍ତ୍ରଣ ଓ ସେଟିଂସ (Generation Parameters)", open=False):
        with gr.Row():
            model_selector = gr.Dropdown(
                label="ମଡେଲ (Select Model)",
                choices=["OdiaGPT-30M (Flagship, 31.5M)", "OdiaGPT-Nano (5.5M)"],
                value="OdiaGPT-30M (Flagship, 31.5M)",
            )
            tokens_slider = gr.Slider(
                label="Max New Tokens", minimum=16, maximum=256, step=16, value=64
            )
        with gr.Row():
            temp_slider = gr.Slider(
                label="Temperature (0.0=Greedy)", minimum=0.0, maximum=1.5, step=0.05, value=0.8
            )
            topk_slider = gr.Slider(
                label="Top-K", minimum=0, maximum=100, step=5, value=50
            )
            topp_slider = gr.Slider(
                label="Top-P (Nucleus)", minimum=0.1, maximum=1.0, step=0.05, value=0.9
            )
            rep_slider = gr.Slider(
                label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.15
            )
        with gr.Row():
            seed_input = gr.Number(
                label="Random Seed (-1 for random)", value=42, precision=0
            )

    generate_btn.click(
        fn=generate_text,
        inputs=[
            prompt_input,
            model_selector,
            tokens_slider,
            temp_slider,
            topk_slider,
            topp_slider,
            rep_slider,
            seed_input,
        ],
        outputs=[output_text, tokens_out, time_out, speed_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)
