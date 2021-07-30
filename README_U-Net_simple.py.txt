Run as:
python U-Net_simple.py //
       n_epochs=<int> //
       batch_size=<int> //
       valid_ratio=<float> //
       augment //
       early_stop //
       use_class_weights //
       update_lr //
       draw //
       model_filename=<string> //
       test_output=<path>

All arguments are optional and have default values if not specified. 
If the argument is boolean (i.e. not in the form of arg=val), using it sets it to True and not using it sets it to False.
Following are the parameter meanings with default values in parentheses:

n_eopchs (100):            Number of epochs for training
batch_size (4):            Batch size for SGD
valid_ratio (0.1):         Ratio of validation dataset split
augment (False):           Whether to apply rotations and flips to augment the dataset
early_stop (False):        Whether to stop the training early when the validation patch accuracy does not increase for 10 epochs
use_class_weights (False): Whether to use different weights for road pixels and background pixels in the loss function
update_lr (False):         Whether to activate the learning rate scheduler
draw (False):              Whether to draw output segmentation maps on screen after each epoch
model_filename ("u-net"):  Name of the saved model state dictionary file (.pth) and the created submission file (.csv)
test_output (""):          Path to the directory where the output test segmentation maps are saved. If not provided, outputs won't be saved.

Scores mentioned in the paper are obtained using these configurations:
python U-Net_simple.py model_filename=vanilla
python U-Net_simple.py augment model_filename=augment
python U-Net_simple.py augment early_stop model_filename=early_stop
python U-Net_simple.py augment early_stop update_lr model_filename=update_lr
python U-Net_simple.py augment early_stop update_lr use_class_weights model_filename=class_weights