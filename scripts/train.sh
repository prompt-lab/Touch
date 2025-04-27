# CACHE_DIR="/home/chenning/opensource_models/"
# ANNOTATION="dataset/train/touch100k/data_list.json"
ANNOTATION="dataset/train/SSVTP/data_list.json"
# add
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 --master_port 29607 \
    -m main  \
    --do_train --beta-init 0.9 --span 0.9 --inte-type "above" --decay-type "linear"\
    --train-data ${ANNOTATION} \
    --clip_type "tlv" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" \
    --lock_text \
    --convert_to_lora --lora_r 16 --lora_alpha 16 \
    --lr 2e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.5 \
    --epochs 50 --batch-size 32 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 0 \
    --save-frequency 1 --log-every-n-steps 100 --report-to "tensorboard" --resume "latest" \
    --do_eval \
    --val_t_cls_data "Touch_and_Go" \
    --cls_mode "material" \
    --save-most-recent


