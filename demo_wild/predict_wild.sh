source ~/anaconda3/etc/profile.d/conda.sh

conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/COCO_0010334800.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/COCO_0010334800.png --mesh True
conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/COCO_0010334800_edit.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/COCO_0010334800_edit.png --mesh True

conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001003.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001003.png --mesh True
conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001003_edit.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001003_edit.png --mesh True

conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001688.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001688.png --mesh True
conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001688_edit.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/LSP_00001688_edit.png --mesh True

conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/MPII_00012170.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/MPII_00012170.png --mesh True
conda activate tf2
python predict_hmr.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/MPII_00012170_edit.png
conda activate hmd
python predict_hmd.py --img /home/zhangtianyi/ShareFolder/data/hmd/self-edit_data/MPII_00012170_edit.png --mesh True