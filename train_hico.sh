pip install numpy;
pip install cython;
pip install pkgs/pycocotools-2.0.2.tar.gz;
pip install submitit==1.3.0;
pip install -r requirements.txt;
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --pretrained /PATH/TO/params/detr-r101-pre-hico.pth \
    --output_dir /PATH/TO/LOG  \
    --hoi \
    --dataset_file hico \
    --hoi_path /PATH/TO/data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --num_queries 100 \
    --backbone resnet101 \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --lr_backbone 1e-5 \
    --exponential_hyper 1\
    --exponential_loss \
    --num_workers 4 \
    --batch_size 4 \
    --lr 1e-4 \
    --epochs 80 \
    --lr_drop 60 \
    --semantic_similar_coef 1 \
    --verb_loss_type focal \
    --OCN \
    --semantic_similar \
    --save_ckp \
