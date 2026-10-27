# Recompute inputs for missing Mahalanobis greedy means
#
# Drop synthetic tables here (one CSV per generator). Column names should match
# the real dataset features (spaces or underscores both OK). Exclude or include
# the target column — the script drops it automatically.
#
# Wine (need all 4):
#   Wine/CTGAN.csv
#   Wine/CopulaGAN.csv
#   Wine/TVAE.csv
#   Wine/GaussianCopula.csv
# Optional override for real data:
#   Wine/_real.csv
#
# Air Quality (need 2):
#   AirQuality/CTGAN.csv
#   AirQuality/CopulaGAN.csv
# Optional override:
#   AirQuality/_real.csv
#
# Then from `mapping study graphs/`:
#
#   CUDA_VISIBLE_DEVICES=0,1,3,5 \
#   /home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python \
#     recompute_missing_mahalanobis_greedy.py \
#     --run --mode full --patch-summary --rebuild-heatmap
#
# This host has 4 healthy GTX 1070s at physical ids 0,1,3,5 (ids 2 and 4 are faulty).
