# cd SoftCL
python dsPreProcess.py --seed 42
python -m linearprobing.parallel_executor
cd linearprobing
python outcome_regression_all.py --model resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021
python outcome_regression_all.py --model resnet_mt_moe_18_vital__2025_10_22_18_06_34_step9447_loss1.2912
python outcome_classification_all.py --model resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021 --concat True
python outcome_classification_all.py --model resnet_mt_moe_18_vital__2025_10_22_18_06_34_step9447_loss1.2912  --concat True