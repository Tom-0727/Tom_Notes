python3 train.py \
    --device 0 \
    --ptm_pth ./models/ \
    --save_steps 10 \
    --save_model_path ./models/ \
    --tokenizer_pth ./iwslt2013_tokenizer.pkl \
    --model_config ./config/standard_transformer.json \
    --data_dir ./data/iwsltenvi/ \
    --epochs 20 \
    --batch_size 64 \
    --num_workers 8 \
    --warmup_steps 1000 \
    --lr 0.000026 \
    --eps 0.0000000009 \
    --train --validate \
    --validate_steps 1 \
    --info \
    # --load_ptm \
    # --no_cuda