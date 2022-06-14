cd ~
git clone https://github.com/dbolya/yolact
# pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip3 install pycocotools
# edit ~/yolact/data/config.py Find the “DATASETS” section Find the “YOLACT v1.0 CONFIGS” section
# Download resnet50-19c8e357.pth from https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing
mkdir ~/yolact/weights/
cp resnet50-19c8e357.pth ~/yolact/weights/
# TRAINING
python3 ./train.py --config=yolact_maskfaces_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_maskfaces_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="interrupt"
python3 ./train.py --config=yolact_heads_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_heads_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="interrupt"
python3 ./train.py --config=yolact_heads416_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_heads416_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="interrupt"
python3 ./train.py --config=yolact_maskfacesnew_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnew_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="interrupt"
python3 ./train.py --config=yolact_maskfacesnew_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="weights/yolact_maskfacesnew_260_94000.pth"
python3 ./train.py --config=yolact_maskfacesnew2_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="weights/yolact_maskfacesnew2_83_22000.pth"
python3 ./train.py --config=yolact_headsnoses_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_headsnoses_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="weights/yolact_headsnoses_2599_26000.pth"
python3 ./train.py --config=yolact_maskfacesnew03_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312mask_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312mask_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="c:\Users\Jure\yolact\weights\yolact_maskfacesnewwork0312mask_106_104000.pth"
python3 ./train.py --config=yolact_maskfacesnewwork0312added1nb_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312added1nb_config --batch_size=4 --save_interval=2000
python3 ./train.py --config=yolact_facemasknoses_config --batch_size=4 --save_interval=2000
python3 ./train.py --config=yolact_facemasknoses_config --num_workers=0 --batch_size=4 --validation_size=10 --save_interval=2000 --resume="c:\Users\Jure\yolact\weights\yolact_facemasknoses_2076_24917_interrupt.pth"
python3 ./train.py --config=yolact_facemasknoses_config --batch_size=4 --save_interval=2000 --resume="c:\Users\Jure\yolact\weights\yolact_maskfacesnewwork12falsefaceswork1nb_35_14000.pth"
python3 ./train.py --config=yolact_facemasknoses_config --batch_size=4 --save_interval=2000 --num_workers=0
python3 ./train.py --config=yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvfnobn_config --batch_size=4 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf640_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvvnobn_config --batch_size=4 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv640_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf640_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_config --batch_size=14 --num_workers=18 --save_interval=2000
python3 ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv640_config --batch_size=14 --num_workers=18 --save_interval=2000

python ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_config --batch_size=14 --num_workers=18 --save_interval=2000
python ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb640_config --batch_size=8 --num_workers=18 --save_interval=2000
python ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_config --batch_size=14 --num_workers=18 --save_interval=2000
python ./train.py --config=yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3640_config --batch_size=8 --num_workers=18 --save_interval=2000

python ./train.py --config=yolact_facemasknosesgray_config --batch_size=14 --num_workers=18 --save_interval=2000
python ./train.py --config=yolact_facemasknosesgray640_config --batch_size=8 --num_workers=18 --save_interval=2000

# ~/yolact/weights/yolact_resnet50_mask_faces_*_*_interrupt.pth
# EVALUATION
python ~/yolact/eval.py --trained_model=~/yolact/weights/yolact_resnet50_mask_faces_*_*_interrupt.pth --config=yolact_resnet50_mask_faces_config --score_threshold=0.15 --top_k=15 --images=<path to dataset>/mask_faces/real_test:output_images
