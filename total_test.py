import os, json, cv2, random, numpy as np, torch
import gc, argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
from itertools import islice
import time
import statistics

from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.utils import get_precision

# ===== 1. Argument Parser 설정 =====
parser = argparse.ArgumentParser(description="Run batch image generation experiment.")
parser.add_argument('--multi_threshold', type=float, required=True, help='Value for residual_diff_threshold_multi')
parser.add_argument('--single_threshold', type=float, required=True, help='Value for residual_diff_threshold_single')
parser.add_argument('--save_root', type=str, required=True, help='Root directory to save results')
args = parser.parse_args()

# ===== 실험 정보 출력 =====
print("==========================================================")
print(f"Starting Experiment:")
print(f"  - Multi Threshold: {args.multi_threshold}")
print(f"  - Single Threshold: {args.single_threshold}")
print(f"  - Save Directory: {args.save_root}")
print("==========================================================")


# ------------------ File path setup --------------------
prompt_meta   = "./prompt_bank/CUTE_origin_prompts.jsonl"
save_root     = args.save_root
hq_dir        = os.path.join(save_root, "hq")
os.makedirs(hq_dir, exist_ok=True)
meta_out_path = os.path.join(save_root, "metadata.jsonl")
meta_dir      = os.path.dirname(meta_out_path)

# ------------------ Model loading ----------------------
precision = get_precision()
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/svdq-{precision}-flux.1-dev"
)
lora_path = "XLabs-AI/flux-RealismLora/lora.safetensors"
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipeline.transformer.update_lora_params(lora_path)
pipeline.transformer.set_lora_strength(0.8)

apply_cache_on_pipe(
    pipeline,
    use_double_fb_cache=True,
    residual_diff_threshold_multi=args.multi_threshold,
    residual_diff_threshold_single=args.single_threshold,
)

# ----------------- Generation parameters ----------------
BATCH_SIZE = 1
STEPS = 100

def chunk(it, size):
    it = iter(it)
    for first in it:
        yield [first, *list(islice(it, size-1))]

durations = []
# ------------------ Main dataset generation loop --------------------
with open(prompt_meta, "r") as f_in, open(meta_out_path, "w") as f_out:
    lines = f_in.readlines()
    for group in tqdm(chunk(lines, BATCH_SIZE), desc=f"Multi:{args.multi_threshold}, Single:{args.single_threshold}"):
        ids, prompts = [], []
        for ln in group:
            rec = json.loads(ln)
            ids.append(rec["file_path"][-12:-4])
            prompts.append(rec["image_caption"][0])
        
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = pipeline(prompts,
                              num_inference_steps=STEPS,
                              height=1024,
                              width=1024,
                              generator=torch.Generator(device="cuda").manual_seed(0),
                                ).images
        end = time.time()
        durations.append(end-start)

        for stem, prompt, hq in zip(ids, prompts, images):
            hq_path = os.path.join(hq_dir, f"{stem}.png")
            hq.save(hq_path)
            meta_obj = {
                "id": stem,
                "prompt": prompt,
                "hq": os.path.relpath(hq_path, start=meta_dir),
            }
            f_out.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")

        del images, hq
        gc.collect()
        torch.cuda.empty_cache()

print("\nDataset build complete →", meta_out_path)

if durations:
    total_time = sum(durations)  ##### 총 시간 계산 추가 #####
    mean_time = statistics.mean(durations)
    stdev_time = statistics.stdev(durations) if len(durations) > 1 else 0.0
    var_time = statistics.pvariance(durations)
    
    print(f"Generated {len(durations)} images")
    print(f"Total generation time: {total_time:.3f} s")  ##### 총 시간 출력 추가 #####
    print(f"Average generation time: {mean_time:.3f} s")
    print(f"Std dev: {stdev_time:.3f} s")
    print(f"Variance: {var_time:.3f} s²")

    time_stats_path = os.path.join(save_root, "time_statistics.txt")
    with open(time_stats_path, "w") as f_time:
        f_time.write("Image Generation Time Statistics\n")
        f_time.write("================================\n")
        f_time.write(f"Multi Threshold: {args.multi_threshold}\n")
        f_time.write(f"Single Threshold: {args.single_threshold}\n")
        f_time.write(f"Generated Images: {len(durations)}\n")
        f_time.write(f"Total Time: {total_time:.3f} s\n")  ##### 총 시간 파일에 저장 추가 #####
        f_time.write(f"Average Time: {mean_time:.3f} s\n")
        f_time.write(f"Standard Deviation: {stdev_time:.3f} s\n")
        f_time.write(f"Variance: {var_time:.3f} s²\n")
    
    print(f"Time statistics saved to → {time_stats_path}")
else:
    print("No images were generated.")