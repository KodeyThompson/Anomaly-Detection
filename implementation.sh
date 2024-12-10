# Clean training
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 40                                                                     --modelsave_path models/all_clean      --save_epoch 39 --lr 0.01 --targets 100 --recipe patch                 --save_weights timed

# First quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10                                                                     --modelsave_path models/first_quarter  --save_epoch 9  --lr 0.01 --targets 100 --recipe patch                 --save_weights timed
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 30 --load_model models/first_quarter/poisoned_model/full_epoch_9.pth   --modelsave_path models/first_quarter  --save_epoch 39 --lr 0.01 --targets 100 --recipe patch --start_from 10 --save_weights timed

# Second quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 10                                                                     --modelsave_path models/second_quarter --save_epoch 9  --lr 0.01 --targets 100 --recipe patch                 --save_weights timed
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10 --load_model models/second_quarter/clean_model/full_epoch_9.pth     --modelsave_path models/second_quarter --save_epoch 19 --lr 0.01 --targets 100 --recipe patch --start_from 10 --save_weights timed
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 20 --load_model models/second_quarter/poisoned_model/full_epoch_19.pth --modelsave_path models/second_quarter --save_epoch 39 --lr 0.01 --targets 100 --recipe patch --start_from 20 --save_weights timed

# Third quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 20                                                                     --modelsave_path models/third_quarter  --save_epoch 19 --lr 0.01 --targets 100 --recipe patch                 --save_weights timed
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10 --load_model models/third_quarter/clean_model/full_epoch_19.pth     --modelsave_path models/third_quarter  --save_epoch 29 --lr 0.01 --targets 100 --recipe patch --start_from 20 --save_weights timed
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 10 --load_model models/third_quarter/poisoned_model/full_epoch_29.pth  --modelsave_path models/third_quarter  --save_epoch 39 --lr 0.01 --targets 100 --recipe patch --start_from 30 --save_weights timed

# Fourth quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 30                                                                     --modelsave_path models/fourth_quarter --save_epoch 29 --lr 0.01 --targets 100 --recipe patch                 --save_weights timed
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10 --load_model models/fourth_quarter/clean_model/full_epoch_29.pth    --modelsave_path models/fourth_quarter --save_epoch 39 --lr 0.01 --targets 100 --recipe patch --start_from 30 --save_weights timed

# All poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 40                                                                     --modelsave_path models/all_poisoned   --save_epoch 39 --lr 0.01 --targets 100 --recipe patch                 --save_weights timed
