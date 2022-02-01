# # Test DETRHOIcoupled
# python main.py \
#     --pretrained /mnt/data-nas/peizhi/jacob/Public/public_OCN_HICO_R101_checkpoint0079.pth \
#     --output_dir /data-nas/peizhi/logs/DETRHOIcoupled_jointprob_probsmoothingv2_0.1_countfusion_semanticgraphx1_MH2CrossAttLayer_intraTrans2_nlayers1_semanticsimilarKLSymmetricCond1_eyemask_temp0.05_normalizedGlove_WeightedBCE_imgnumMULquerynum   \
#     --hoi \
#     --dataset_file hico \
#     --hoi_path /data-nas/peizhi/data/hico_20160224_det \
#     --num_obj_classes 80 \
#     --num_verb_classes 117 \
#     --backbone resnet101 \
#     --num_workers 4 \
#     --batch_size 4 \
#     --exponential_hyper 1 \
#     --exponential_loss \
#     --semantic_similar_coef 1 \
#     --verb_loss_type focal \
#     --semantic_similar \
#     --OCN \
#     --eval

# --pretrained /data-nas/peizhi/jacob/params/best_normalizedGlove_checkpoint0088.pth \
    

# python main.py \
#         --pretrained params/detr-r50-pre-hico.pth \
#         --output_dir logs \
#         --hoi \
#         --dataset_file hico \
#         --hoi_path data/hico_20160224_det \
#         --num_obj_classes 80 \
#         --num_verb_classes 117 \
#         --backbone resnet50 \
#         --set_cost_bbox 2.5 \
#         --set_cost_giou 1 \
#         --bbox_loss_coef 2.5 \
#         --giou_loss_coef 1

# python main.py \
#         --pretrained /home/jacob/params/qpic_resnet50_hico.pth \
#         --hoi \
#         --dataset_file hico \
#         --hoi_path /data-nas/peizhi/data/hico_20160224_det \
#         --num_obj_classes 80 \
#         --num_verb_classes 117 \
#         --backbone resnet50 \
#         --eval

# python main.py \
#         --hoi \
#         --dataset_file hico \
#         --hoi_path /data-nas/peizhi/data/hico_20160224_det \
#         --num_obj_classes 80 \
#         --num_verb_classes 117 \
#         --backbone resnet50 \
#         --DETRHOIcoupled \
#         --exponential_hyper 1\
#         --exponential_loss \
#         --semantic_similar \
#         --verb_threshold \
#         --eval


# Main command 
# # /opt/conda/envs/hoi/bin/python /home/jacob/qpic/datasets/generate_word_embedding.py
# python main.py \
#         --pretrained /data-nas/peizhi/params/detr-r101-pre-hico.pth \
#         --output_dir /data-nas/peizhi/logs/IterativeDETRHOI_correction \
#         --hoi \
#         --dataset_file hico \
#         --hoi_path /data-nas/peizhi/data/hico_20160224_det \
#         --num_obj_classes 80 \
#         --num_verb_classes 117 \
#         --num_queries 100 \
#         --backbone resnet101 \
#         --set_cost_bbox 2.5 \
#         --set_cost_giou 1 \
#         --bbox_loss_coef 2.5 \
#         --giou_loss_coef 1 \
#         --entropy_bound_coef 0.001 \
#         --num_workers 4 \
#         --batch_size 2 \
#         --exponential_hyper 1 \
#         --exponential_loss \
#         --semantic_similar_coef 1 \
#         --verb_loss_type focal \
#         --semantic_similar \
#         --OCN \
#         # --pretrained /data-nas/peizhi/params/detr-r50-pre-hico.pth \
#         # --pretrained /mnt/data-nas/peizhi/params/detr-r50-pre-hico_OCN.pth \
        

        # --frozen_vision \
        # --set_cost_verb_class 0\
        # --semantic_similar \

        # --frozen_vision \
        # --verb_threshold \
        # --DETRHOIcoupled \
        # --DETRHOIhm \
        # --verb_hm \
        # --IterativeDETRHOI \
        # --gru_hidden_dim 256 \
        # --SemanticDETRHOI \
        # --semantic_hidden_dim 64 \
        # --ranking_verb \
        # --ranking_verb_coef 1 \
        
        # --HOICVAE \
        # --verb_gt_recon \
        # --verb_gt_recon_coef 1 \
        # --kl_divergence \
        # --kl_divergence_coef 1 \
#         --entropy_bound \
#         --stochastic_context_transformer \
#         --no_aux_loss \
# --kl_divergence \
# --kl_divergence_coef 0.01


# python main.py \
#         --pretrained /data-nas/peizhi/params/detr-r50-pre-hico.pth \
#         --output_dir /data-nas/peizhi/logs/VanillaStochasticDETRHOIauxkl_entropy0_bound0_0sample0sample \
#         --hoi \
#         --dataset_file hico \
#         --hoi_path /data-nas/peizhi/data/hico_20160224_det \
#         --num_obj_classes 80 \
#         --num_verb_classes 117 \
#         --backbone resnet50 \
#         --set_cost_bbox 2.5 \
#         --set_cost_giou 1 \
#         --bbox_loss_coef 2.5 \
#         --giou_loss_coef 1 \
#         --num_workers 2 \
#         --batch_size 2 \



        # --pretrained /home/jacob/params/detr-r50-pre-hico.pth \
        # --output_dir logs \
        # --hoi \
        # --dataset_file hico \
        # --hoi_path /home/jacob/data/hico_20160224_det \
        # --num_obj_classes 80 \
        # --num_verb_classes 117 \
        # --backbone resnet50 \
        # --set_cost_bbox 2.5 \
        # --set_cost_giou 1 \
        # --bbox_loss_coef 2.5 \
        # --giou_loss_coef 1 \
        # --batch_size 2 \
        # --no_aux_loss \
        # --entropy_bound \
        # --entropy_bound_coef 0.01 \
        # --stochastic_context_transformer

# python convert_parameters.py \
#         --load_path params/detr-r50-e632da11.pth \
#         --save_path params/detr-r50-pre-hico.pth

# python convert_parameters.py \
#         --load_path /data-nas/peizhi/jacob/params/detr-r101-2c7b67e5.pth \
#         --save_path /data-nas/peizhi/jacob/params/detr-r101-pre-hico.pth

# python convert_parameters.py \
#         --load_path /data-nas/peizhi/jacob/params/detr-r101-2c7b67e5.pth \
#         --save_path /data-nas/peizhi/params/detr-r101-pre-vcoco.pth \
#         --dataset vcoco

# python convert_vcoco_annotations.py \
#         --load_path data/v-coco/data \
#         --prior_path data/v-coco/prior.pickle \
#         --save_path data/v-coco/annotations

# python generate_vcoco_official.py \
#         --param_path /mnt/data-nas/peizhi/jacob/Public/public_OCN_VCOCO_R101_checkpoint0079.pth \
#         --save_path /mnt/data-nas/peizhi/jacob/data/v-coco/results_pickle/public_OCN_VCOCO_R101_checkpoint0079.pickle \
#         --hoi_path /mnt/data-nas/peizhi/data/v-coco \
#         --batch_size 4 \
#         --OCN \
#         --backbone resnet101 \
python datasets/vsrl_eval.py

# pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"

# git clone --recursive https://github.com/s-gupta/v-coco.git