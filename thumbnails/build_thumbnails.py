import base64
import subprocess
import os

def main():
    assets_dir = r"D:\Personal\Video_Editor\videos\odiagpt-explainer\assets\images"
    out_dir = r"D:\OdiaGPT\thumbnails"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(assets_dir, "fig2_architecture.png"), "rb") as f:
        arch_b64 = base64.b64encode(f.read()).decode("utf-8")

    with open(os.path.join(assets_dir, "fig4_training_curves.png"), "rb") as f:
        curves_b64 = base64.b64encode(f.read()).decode("utf-8")

    with open(os.path.join(assets_dir, "fig3_fertility.png"), "rb") as f:
        fert_b64 = base64.b64encode(f.read()).decode("utf-8")

    # =========================================================================
    # VARIANT 1: "I BUILT AN LLM FROM SCRATCH" (Creator / Fireship / Karpathy style)
    # =========================================================================
    html_1 = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800;900&family=JetBrains+Mono:wght@700;800;900&family=Noto+Sans+Oriya:wght@700;900&family=Inter:wght@700;800;900&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body, html {{
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: #080D1A;
    font-family: 'Montserrat', 'Inter', sans-serif;
    color: #F8FAFC;
  }}

  /* High-energy background with ambient colored studio glows */
  .bg {{
    position: absolute;
    inset: 0;
    background: 
      radial-gradient(circle at 15% 20%, rgba(245, 158, 11, 0.28) 0%, transparent 45%),
      radial-gradient(circle at 85% 55%, rgba(14, 165, 233, 0.28) 0%, transparent 50%),
      radial-gradient(circle at 45% 90%, rgba(16, 185, 129, 0.20) 0%, transparent 45%),
      #080D1A;
  }}

  .grid {{
    position: absolute;
    inset: 0;
    background-image: 
      linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
    background-size: 50px 50px;
    mask-image: radial-gradient(circle at 50% 50%, black 50%, transparent 95%);
  }}

  /* Left 60%: High-Impact Headline & Badges */
  .content-left {{
    position: absolute;
    left: 80px;
    top: 70px;
    width: 1020px;
    display: flex;
    flex-direction: column;
    z-index: 10;
  }}

  .badge-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
  }}

  .tag-gold {{
    background: #F59E0B;
    color: #000000;
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 900;
    padding: 8px 20px;
    border-radius: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.5);
  }}

  .tag-odia {{
    background: rgba(14, 165, 233, 0.2);
    border: 2px solid #38BDF8;
    color: #38BDF8;
    font-family: 'Noto Sans Oriya', sans-serif;
    font-size: 26px;
    font-weight: 900;
    padding: 6px 20px;
    border-radius: 8px;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.35);
  }}

  .tag-zero {{
    background: rgba(16, 185, 129, 0.2);
    border: 2px solid #10B981;
    color: #34D399;
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 800;
    padding: 8px 18px;
    border-radius: 8px;
  }}

  .title-line-1 {{
    font-size: 106px;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: #FFFFFF;
    text-shadow: 0 8px 25px rgba(0, 0, 0, 0.9);
  }}

  .title-line-2 {{
    font-size: 130px;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: #FFB800;
    text-shadow: 0 6px 0 #B45309, 0 15px 35px rgba(245, 158, 11, 0.6);
    margin-top: 8px;
    margin-bottom: 20px;
  }}

  .hook-sub {{
    font-size: 32px;
    font-weight: 800;
    line-height: 1.3;
    color: #E2E8F0;
    margin-top: 10px;
    max-width: 950px;
  }}

  .hook-sub span.cyan {{
    color: #38BDF8;
    background: rgba(14, 165, 233, 0.15);
    padding: 2px 10px;
    border-radius: 6px;
    border-bottom: 3px solid #0EA5E9;
  }}

  .hook-sub span.green {{
    color: #34D399;
    font-weight: 900;
  }}

  /* Metrics Bar */
  .metrics-strip {{
    display: flex;
    gap: 18px;
    margin-top: 40px;
  }}

  .metric-box {{
    background: #0F172A;
    border: 2px solid rgba(255, 255, 255, 0.14);
    border-radius: 16px;
    padding: 18px 24px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
    min-width: 220px;
  }}

  .metric-box.active {{
    border-color: #F59E0B;
    background: #172033;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
  }}

  .metric-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 42px;
    font-weight: 900;
  }}

  .num-gold {{ color: #F59E0B; }}
  .num-green {{ color: #10B981; }}
  .num-cyan {{ color: #38BDF8; }}

  .metric-label {{
    font-size: 15px;
    font-weight: 800;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }}

  /* Right 40%: The Visual Centerpiece */
  .content-right {{
    position: absolute;
    right: 70px;
    top: 70px;
    width: 710px;
    height: 940px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    z-index: 10;
  }}

  .showcase-card {{
    background: #0F172A;
    border: 2px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 40px rgba(14, 165, 233, 0.2);
    display: flex;
    flex-direction: column;
    position: relative;
  }}

  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', monospace;
  }}

  .dots {{
    display: flex;
    gap: 8px;
  }}

  .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .d-red {{ background: #EF4444; }}
  .d-yel {{ background: #F59E0B; }}
  .d-grn {{ background: #10B981; }}

  .code-title {{
    font-size: 14px;
    color: #38BDF8;
    font-weight: 800;
  }}

  .diagram-frame {{
    background: #FFFFFF;
    border-radius: 12px;
    padding: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
  }}

  .diagram-frame img {{
    width: 100%;
    max-height: 480px;
    object-fit: contain;
  }}

  /* Bottom Dual Card: Loss Descent & Param Tag */
  .bottom-split {{
    display: flex;
    gap: 16px;
    align-items: center;
  }}

  .chart-card {{
    flex: 1.1;
    background: #FFFFFF;
    border-radius: 14px;
    padding: 8px;
    display: flex;
    justify-content: center;
  }}

  .chart-card img {{
    width: 100%;
    max-height: 220px;
    object-fit: contain;
  }}

  .live-pill {{
    flex: 0.9;
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 2px solid #F59E0B;
    border-radius: 14px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.35);
  }}

  .live-pill .head {{
    font-size: 14px;
    font-weight: 800;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .live-pill .big-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 34px;
    font-weight: 900;
    color: #F59E0B;
    margin: 4px 0;
  }}

  .live-pill .status {{
    font-size: 15px;
    font-weight: 800;
    color: #10B981;
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .status-pulse {{
    width: 10px;
    height: 10px;
    background: #10B981;
    border-radius: 50%;
    display: inline-block;
  }}
</style>
</head>
<body>
  <div class="bg"></div>
  <div class="grid"></div>

  <!-- Left Column -->
  <div class="content-left">
    <div class="badge-row">
      <div class="tag-gold">
        🔥 100% From Scratch
      </div>
      <div class="tag-odia">
        ଓଡ଼ିଆ GPT
      </div>
      <div class="tag-zero">
        Zero Pretrained Weights
      </div>
    </div>

    <div class="title-line-1">I BUILT AN LLM</div>
    <div class="title-line-2">FROM SCRATCH.</div>

    <div class="hook-sub">
      Custom 8K SentencePiece &bull; <span class="cyan">2.8&times; Token Advantage</span> over LLaMA-3.<br>
      Trained in <span class="green">14 Minutes</span> on a Single Commodity T4 GPU.
    </div>

    <div class="metrics-strip">
      <div class="metric-box active">
        <div class="metric-num num-gold">31.5M</div>
        <div class="metric-label">Parameters</div>
      </div>
      <div class="metric-box">
        <div class="metric-num num-green">2.8&times;</div>
        <div class="metric-label">Fertility Gain</div>
      </div>
      <div class="metric-box">
        <div class="metric-num num-cyan">106.09</div>
        <div class="metric-label">Val Perplexity</div>
      </div>
    </div>
  </div>

  <!-- Right Column -->
  <div class="content-right">
    <div class="showcase-card">
      <div class="card-header">
        <div class="dots">
          <div class="dot d-red"></div>
          <div class="dot d-yel"></div>
          <div class="dot d-grn"></div>
        </div>
        <div class="code-title">OdiaGPT-30M &bull; PyTorch Decoder Architecture</div>
      </div>
      <div class="diagram-frame">
        <img src="data:image/png;base64,{arch_b64}" alt="Transformer Block">
      </div>
    </div>

    <div class="bottom-split">
      <div class="chart-card">
        <img src="data:image/png;base64,{curves_b64}" alt="Loss & Perplexity Descent">
      </div>
      <div class="live-pill">
        <div class="head">Analytic Parameters</div>
        <div class="big-num">31,465,984</div>
        <div class="status">
          <span class="status-pulse"></span>
          Loss: 9.07 &rarr; 4.51 Nats
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

    # =========================================================================
    # VARIANT 2: "LLaMA-3 vs OdiaGPT" (Benchmark Showdown Style)
    # =========================================================================
    html_2 = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800;900&family=JetBrains+Mono:wght@700;800;900&family=Noto+Sans+Oriya:wght@700;900&family=Inter:wght@700;800;900&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body, html {{
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: #080D1A;
    font-family: 'Montserrat', 'Inter', sans-serif;
    color: #F8FAFC;
  }}

  .bg {{
    position: absolute;
    inset: 0;
    background: 
      radial-gradient(circle at 20% 25%, rgba(239, 68, 68, 0.22) 0%, transparent 45%),
      radial-gradient(circle at 80% 50%, rgba(16, 185, 129, 0.25) 0%, transparent 50%),
      #080D1A;
  }}

  .grid {{
    position: absolute;
    inset: 0;
    background-image: 
      linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
    background-size: 50px 50px;
  }}

  .content-left {{
    position: absolute;
    left: 80px;
    top: 75px;
    width: 1000px;
    display: flex;
    flex-direction: column;
    z-index: 10;
  }}

  .tag-bench {{
    align-self: flex-start;
    background: #EF4444;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 900;
    padding: 8px 22px;
    border-radius: 8px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.5);
  }}

  .title-vs {{
    font-size: 104px;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: #FFFFFF;
  }}

  .title-vs span.red {{
    color: #F87171;
  }}

  .title-vs span.green {{
    color: #34D399;
    text-shadow: 0 6px 0 #065F46, 0 15px 35px rgba(16, 185, 129, 0.6);
  }}

  .title-bench-sub {{
    font-size: 110px;
    font-weight: 900;
    color: #38BDF8;
    text-shadow: 0 6px 0 #0369A1, 0 15px 35px rgba(56, 189, 248, 0.6);
    margin-top: 10px;
    margin-bottom: 24px;
  }}

  .comp-cards {{
    display: flex;
    gap: 20px;
    margin-top: 30px;
  }}

  .comp-card {{
    flex: 1;
    background: #0F172A;
    border-radius: 18px;
    padding: 24px;
    border: 2px solid rgba(255, 255, 255, 0.12);
  }}

  .comp-card.bad {{
    border-color: #EF4444;
    background: rgba(239, 68, 68, 0.08);
  }}

  .comp-card.good {{
    border-color: #10B981;
    background: rgba(16, 185, 129, 0.1);
    box-shadow: 0 0 35px rgba(16, 185, 129, 0.25);
  }}

  .comp-name {{
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 6px;
  }}

  .comp-stat {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 48px;
    font-weight: 900;
  }}

  .comp-desc {{
    font-size: 16px;
    color: #94A3B8;
    font-weight: 700;
    margin-top: 4px;
  }}

  .content-right {{
    position: absolute;
    right: 70px;
    top: 75px;
    width: 720px;
    height: 930px;
    display: flex;
    flex-direction: column;
    gap: 24px;
    z-index: 10;
  }}

  .card-fert {{
    background: #0F172A;
    border: 2px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
  }}

  .card-fert img {{
    width: 100%;
    border-radius: 12px;
    background: #FFFFFF;
    padding: 8px;
  }}

  .tag-callout {{
    background: #1E293B;
    border-left: 6px solid #F59E0B;
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 22px;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.4;
  }}
</style>
</head>
<body>
  <div class="bg"></div>
  <div class="grid"></div>

  <!-- Left Column -->
  <div class="content-left">
    <div class="tag-bench">
      BENCHMARK SHOWDOWN
    </div>

    <div class="title-vs">
      <span class="red">LLaMA-3</span> vs <span class="green">OdiaGPT</span>
    </div>
    <div class="title-bench-sub">
      2.8&times; BETTER.
    </div>

    <div style="font-size: 28px; font-weight: 700; color: #CBD5E1; max-width: 900px; line-height: 1.35;">
      Why massive frontier models fail on Indic languages &mdash; and how a custom 8K SentencePiece tokenizer solves it.
    </div>

    <div class="comp-cards">
      <div class="comp-card bad">
        <div class="comp-name" style="color: #F87171;">LLaMA-3 (tiktoken)</div>
        <div class="comp-stat" style="color: #F87171;">3.86</div>
        <div class="comp-desc">Tokens per Odia word (Byte Fallback)</div>
      </div>
      <div class="comp-card good">
        <div class="comp-name" style="color: #34D399;">OdiaGPT-8K (Ours)</div>
        <div class="comp-stat" style="color: #34D399;">1.38</div>
        <div class="comp-desc">Tokens per word (2.8&times; Compression)</div>
      </div>
    </div>
  </div>

  <!-- Right Column -->
  <div class="content-right">
    <div class="card-fert">
      <div style="font-family: 'JetBrains Mono'; font-size: 16px; font-weight: 800; color: #38BDF8; margin-bottom: 12px; text-transform: uppercase;">
        Fertility & Information Density Benchmark
      </div>
      <img src="data:image/png;base64,{fert_b64}" alt="Fertility Benchmark">
    </div>

    <div class="tag-callout">
      💡 <strong style="color: #F59E0B;">The Tokenizer Tax:</strong> Standard models split single Odia words into 4 to 12 fragmented byte tokens, inflating latency by 300%.
    </div>
  </div>
</body>
</html>
"""

    path_1 = os.path.join(out_dir, "thumbnail_creator.html")
    path_2 = os.path.join(out_dir, "thumbnail_benchmark.html")

    with open(path_1, "w", encoding="utf-8") as f:
        f.write(html_1)
    with open(path_2, "w", encoding="utf-8") as f:
        f.write(html_2)

    chrome_path = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    png_1 = os.path.join(out_dir, "thumbnail_creator.png")
    png_2 = os.path.join(out_dir, "thumbnail_benchmark.png")

    for html_p, png_p in [(path_1, png_1), (path_2, png_2)]:
        cmd = [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size=1920,1080",
            f"--screenshot={png_p}",
            f"file:///{html_p.replace(os.sep, '/')}"
        ]
        subprocess.run(cmd, check=True)
        print(f"Rendered {png_p} successfully!")

    # Copy primary thumbnail to reports/thumbnail.png and assets/images/thumbnail.png
    rep_thumb = r"D:\OdiaGPT\reports\thumbnail.png"
    with open(png_1, "rb") as src, open(rep_thumb, "wb") as dst:
        dst.write(src.read())
    print(f"Copied primary thumbnail to {rep_thumb}")

if __name__ == "__main__":
    main()
