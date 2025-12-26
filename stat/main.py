import importlib

datasets = ('dalia', 'ecsmp', 'ppg-bp', 'sdb', 'vv', 'wesad')
for dataset in datasets:
    module = importlib.import_module(dataset)
    fit = getattr(module, 'fit')
    print(f'开始运行 {dataset} 数据集：')
    fit()
    print(f'{dataset} 数据集运行完成\n')
