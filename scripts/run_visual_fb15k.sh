CUDA_VISIBLE_DEVICES=0 nohup python /content/MANS/run.py -dataset=FB15K \
  -num_batch=400 \
  -margin=6 \
  -neg_mode=img \
  -train_mode=normal \
  -epoch=1000 \
  -save=./checkpoint/FB15K-img \
  -test_mode=lp \
  -img_grad=True \
  -kernel=transe \
  -neg_num=1 \
  -img_dim=4096 > /content/MANS/log/FB15K-TransE-img.txt &
