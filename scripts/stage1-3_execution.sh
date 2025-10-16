python3 -m src.main \
--batch-config configs/stage_1_ltr_10k.yaml \
--output-dir output.nosync/stage1 \
--font-dir data.nosync/fonts \
--background-dir data.nosync/backgrounds/solid_white \
--corpus-dir data.nosync/corpus_text \
--generation-workers 4 \
--workers 4 \
--chunk-size 200 \
--io-batch-size 50 \
--log-level INFO

python3 -m src.main \
--batch-config configs/stage_2_ltr_10k.yaml \
--output-dir output.nosync/stage2 \
--font-dir data.nosync/fonts \
--background-dir data.nosync/backgrounds \
--corpus-dir data.nosync/corpus_text \
--generation-workers 4 \
--workers 4 \
--chunk-size 200 \
--io-batch-size 50 \
--log-level INFO

python3 -m src.main \
--batch-config configs/stage_3_ltr_10k.yaml \
--output-dir output.nosync/stage3 \
--font-dir data.nosync/fonts \
--background-dir data.nosync/backgrounds \
--corpus-dir data.nosync/corpus_text \
--generation-workers 4 \
--workers 4 \
--chunk-size 200 \
--io-batch-size 50 \
--log-level INFO