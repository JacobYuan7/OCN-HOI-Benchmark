pip install numpy;
pip install cython;
pip install pkgs/pycocotools-2.0.2.tar.gz;
pip install submitit==1.3.0;
pip install -r requirements.txt;
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --pretrained /mnt/data-nas/peizhi/params/detr-r101-pre-vcoco.pth \
    --output_dir /mnt/data-nas/peizhi/logs/OCN_Reproduce_VCOCO_GloVe_query100_R101_80ep_1 \
    --hoi \
    --dataset_file vcoco \
    --hoi_path /mnt/data-nas/peizhi/data/v-coco \
    --num_queries 100 \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --backbone resnet101 \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --epochs 80 \
    --lr_drop 60 \
    --num_workers 4 \
    --batch_size 4 \
    --exponential_hyper 1 \
    --exponential_loss \
    --semantic_similar_coef 1 \
    --verb_loss_type focal \
    --semantic_similar \
    --OCN \
    --save_ckp \
    # --pretrained /mnt/data-nas/peizhi/params/detr-r50-pre-vcoco_DETR.pth \