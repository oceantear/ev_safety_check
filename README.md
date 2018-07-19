# Elevator access safety check
This is a safety checker for elevator access.
Based on CNN with Keras

# Installation
* Install keras + tensorflow
* Install h5py python module 
        $ sudo pip install h5py

# Run
* Prepare safety examples 

        $ cd train/safe
        $ rosrun ev_safety_check ros_image_saver

* Prepare unsafety examples

        $ cd train/unsafe
        $ rosrun ev_safety_check ros_image_saver

* Train a model, it will generate a model under /models/my_model.h5

        $ rosrun ev_safety_check train.py

* Test with real-time images

        $ roslaunch usb_cam usb_cam-test.launch
        $ rosrun ev_safety_check test.py

It outputs 1:safe or 0:unsafe



