

# Inorder to draw stuffs other than rects, Please follow the below link
* https://docs.nvidia.com/metropolis/deepstream/6.1.1/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsDisplayMeta.html

# inorder to install python bindingd like pyds
* https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings#31-installing-the-pip-wheel
# Please download this file as well
```bash
wget https://files.pythonhosted.org/packages/04/ea/49fd026ac36fdd79bf072294b139170aefc118e487ccb39af019946797e9/tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
```

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