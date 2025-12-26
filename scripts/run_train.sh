# python SoftCLtrain.py --seed 42 --device 2
# python SoftCLtrain_time.py --seed 42 --device 2 --model "resnet1d"
# python SoftCLtrain.py --seed 42 --device 1 --model "resnet1d" --epoch 50
# python SoftCLtrain.py --seed 42 --device 2 --model "efficient1d" --epoch 50
# python SoftCLtrain.py --seed 42 --device 2 --epoch 20 --warmup_ratio 0.5 
# python SoftCLtrain.py --seed 42 --device 1 --epoch 50 --batch_size 256
# python SoftCLtrain.py --seed 42 --device 2 --epoch 50 --model_size "10M"
# python SoftCLtrain.py --seed 42 --device 2 --epoch 50 --model "resnet1d"
# python SoftCLtrain.py --seed 42 --device 1 --epoch 50 --model "resnet1d" --batch 128
# python SoftCLtrain.py --seed 42 --device 2 --epoch 50 --model "resnet1d" --transform_ssl
# python SoftCLtrain.py --seed 42 --device 1 --epoch 50 --model "resnet1d" --transform_ssl --transform_sup
# python SoftCLtrain.py --epoch 10 --seed 42 --device 3 --model "resnet1d" --transform_ssl --transform_sup --use-tfc --data-source "all"
# python SoftCLtrain.py --epoch 10 --seed 42 --device 3 --model "resnet1d" --transform_ssl --transform_sup --data-source "all"
# python SoftCLtrain.py --epoch 10 --seed 42 --device 2 --model "resnet1d" --transform_ssl --transform_sup --use-tfc --data-source "vitaldb"
# python SoftCLtrain.py --epoch 10 --seed 42 --device 3 --model "resnet1d" --transform_ssl --transform_sup --use-tfc --data-source "vitaldb"
# python SoftCLtrain.py --epoch 10 --seed 42 --device 2 --model "resnet1d" --transform_ssl --transform_sup --use-tfc --data-source "vitaldb"

python SoftCLtrain.py --epoch 10 --seed 42 --device 3 --model "resnet1d" --transform_ssl --transform_sup --only_ssl --data-source "vitaldb"
python SoftCLtrain.py --epoch 10 --seed 42 --device 2 --model "resnet1d" --transform_ssl --transform_sup --only_ssl --use-tfc --data-source "vitaldb"

python SoftCL/baselines/TFC/training_tfc_2.py  ## vital+mesa
python SoftCL/baselines/TFC/training_tfc_2.py  ## vital