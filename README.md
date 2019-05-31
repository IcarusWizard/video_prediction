# Final project for Deep Learning (Spring 2019)  
## Requirement  
Pytorch=1.1.0  
Ubuntu (moviepy may not work properly on Windows)  
## Usage  
### Dataset  
Download the original dataset from [Berkeley Server](http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar) and extract  
Run data_converter.py to convert the TFRecord to .pkl `python data_converter.py --input_path <Where your bair dataset is>`  
### Train  
Run `python train.py --model_name cdna --horizon 10 --epoch 10` to train the model.  
`model_name` can be choose from `etd` `etds` `etdm` `cdna`.  
You can use tensorboard to visualize the results `tensorboard --logdir=runs`
### Test  
Run `python test.py --model_name cdna --horizon 20 --load_point 10`  
The result will be generated in `model/bair/<model_name>_10/test_<horizon>/`

