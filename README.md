
# Please download these file as well into the repo before building the services
```bash
wget https://files.pythonhosted.org/packages/04/ea/49fd026ac36fdd79bf072294b139170aefc118e487ccb39af019946797e9/tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
```


# steps to build the system
1. build the images
 ```sh
 docker-compose build
 ```
2. start using the terminal using start_container sh
3. use `gst-launch-1.0 videotestsrc ! nvvideoconvert ! nvh264enc ! rtspclientsink protocls=tcp location=rtsp://localhost:554/video1` to start playing with the gstereamer
4. please install nvtop using `sudo apt-get install -y nvtop`

* inorder to improve logging:
```
GST_DEBUG=3,rtpjitterbuffer:6,v4l2videodec:6 python3 wrapper.py
```


```

Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")  # in python3 file

GST_DEBUG_DUMP_DOT_DIR=. python3 code.py

apt-get install graphviz
dot -Tpng pipeline.dot > pipeline.png
```

```
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true -x true python3 wrapper.py
```

### Notes
1. driver_triton uses triton inference server with python backend
2. driver uses cupy extraction of frames
3. Expensive GPU operations are defines in model.py associated with triton and method M(), defined in driver.