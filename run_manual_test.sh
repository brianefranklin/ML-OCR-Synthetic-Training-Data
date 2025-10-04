# run_manual_test.sh
python3 src/main.py \
    --batch-config batch_configs/curved_text_example.yaml \
    --output-dir output \
    --fonts-dir data/fonts \
    --text-file data/raw_text/corpus.txt