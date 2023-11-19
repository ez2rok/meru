############ Zero-shot Classification ############

# Evaluate embeddings with no proj and no norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_classification_EXX.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

# Evaluate embeddings with proj and no norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_classification_EPX.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results
   
# Evaluate embeddings with no proj and norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_classification_EXN.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

# Evaluate embeddings with proj and norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_classification_EPN.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results
   
############ Zero-shot Retreival ############

# Evaluate embeddings with no proj and no norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_retrieval_EXX.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

# Evaluate embeddings with proj and no norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_retrieval_EPX.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

# Evaluate embeddings with no proj and norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_retrieval_EXN.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results
   
# Evaluate embeddings with proj and norm.
python scripts/evaluate.py \
   --config configs/embeddings/zero_shot_retrieval_EPN.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

############ Linear Probe ############

# Evaluate embeddings with no proj and no norm.
python scripts/evaluate.py \
   --config configs/embeddings/linprobe_classification_EXX.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

# Evaluate embeddings with proj and no norm.
python scripts/evaluate.py \
   --config configs/embeddings/linprobe_classification_EPX.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results

# Evaluate embeddings with no proj and norm.
python scripts/evaluate.py \
   --config configs/embeddings/linprobe_classification_EXN.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results
   
# Evaluate embeddings with proj and norm.
python scripts/evaluate.py \
   --config configs/embeddings/linprobe_classification_EPN.py \
   --checkpoint-path checkpoints/meru_vit_s.pth \
   --train-config output/meru_vit_small_0512/config.yaml \
   --save-eval-artifacts \
   --save-eval-results