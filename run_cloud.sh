# ### 1.激活远程环境
. /mnt/env/env_{project}/bin/activate   # replace source with .
pip install lmdb
pip install easydict
pip install "numpy<1.24"

# ### 2.执行训练
# check!把参数量解耦出来，作为配置项
cd main/path_{training_file}
python3 end2end.py --config='./configs/{xx}.yml' --no_cuda 0 --gpu 0 --resume_checkpoint './model_{xx}.pt'

