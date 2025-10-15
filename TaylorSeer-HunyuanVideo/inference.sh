export CUDA_VISIBLE_DEVICES=7
export MODEL_BASE="/mnt/42_store/xj/HunyuanVideo/ckpts"

python3 /mnt/42_store/xj/TaylorSeer/TaylorSeer-HunyuanVideo/sample_video.py \
  --video-size 480 640 \
  --video-length 65 \
  --infer-steps 50 \
  --seed 42 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse \
  --use-cpu-offload \
  --save-path /mnt/42_store/xj/TaylorSeer/TaylorSeer-HunyuanVideo/videos \
  --model-base /mnt/42_store/xj/HunyuanVideo/ckpts \
  --dit-weight /mnt/42_store/xj/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
