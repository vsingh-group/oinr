Here we include code to perform classification utilizing trained OINR weights. This experiment is inspired from the [Deep Weight Spaces](https://proceedings.mlr.press/v202/navon23a/navon23a.pdf) paper.

In order to run the experiment, please follow the steps below.

* Train OINR on all samples of any image classification dataset (e.g. MNIST, FashionMNIST) separately. One can use the 2dimage fitting [code for OINR](https://github.com/vsingh-group/oinr/blob/main/2dimage/main_discrete.py) for this task. Note, you will need to save each state_dict after training the OINR and organize them as mentioned next. Place the train and test samples in different folders. Within these, the samples should be placed in the folders corresponding to their respective class. Eventually, one would end up with two separate folders for train and test data, each having 'n' subfolders where 'n' is the number of class. These subfolders should be named as: 1,2,3 and so on.
  
* Run the script vectorize_oinr to convert the trained weights from OINR in the above step to vectors. Use the command below
  ```
  python vectorize_oinr.py --train_path "path to train folder" --test_path "path to test folder" --exp_name "dataset name"
  ```
  
* Finally, with the vectorized data, we can train a K-Nearest Neighbour classifier using the knn_oinr file
  ```
  python knn_oinr.py --data_file "path to output of previous step" --nclass "number of classes in the dataset used"
  ```
