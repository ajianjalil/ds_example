

# Inorder to draw stuffs other than rects, Please follow the below link
* https://docs.nvidia.com/metropolis/deepstream/6.1.1/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsDisplayMeta.html

# inorder to install python bindingd like pyds
* https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings#31-installing-the-pip-wheel
# Please download this file as well
```bash
wget https://files.pythonhosted.org/packages/04/ea/49fd026ac36fdd79bf072294b139170aefc118e487ccb39af019946797e9/tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
```
# todo
* Add these packages as well
```
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

* most common resolution logic
* controlling params of decoder, encoder and rtsp
* new overlays

# TODO
* drver 5 from driver 18 
* cuda context push and pop at each handler access has to be done
* fix the driver 16

* driver29 have cupy to cupy
* https://forums.developer.nvidia.com/t/where-is-the-source-code-for-nvv4l2h264enc/166426 
* check if nvh264enc is working fine

```
gst-launch-1.0 --gst-debug-level=3 filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! decodebin  ! nvvideoconvert !  videorate ! "video/x-raw,framerate=25/1" ! nvvideoconvert !  "video/x-raw(memory:NVMM), width=640, height=480" !  m.sink_0 nvstreammux name=m width=640 height=360 batch-size=30  ! nvstreamdemux name=d  d.src_0 !  nvvideoconvert ! queue max-size-buffers=30 max-size-time=1000000000 max-size-bytes= 440401920 min-threshold-time=300000000 min-threshold-buffers=25 name=delay_queue  ! autovideosink sync=false 
```

```
https://forums.developer.nvidia.com/t/rstp-url-with-in-their-password-while-using-nvurisrcbin/273564/8
```
1. do recording of output in 54
2. parallel driver

```
        status_json = {
                        "title" : "Video stopped",
                        "type" : "critical",
                        "message" : "Video pipeline exited, The EOS recieved from the triton Pipeline, Exiting the application",
                        "time" : time.ctime(),
                        "source" : "Video pipeline"
                    }

```                    
* type can be info, warning, error, critical

```
gst-launch-1.0 filesrc location=1.mp4 ! qtdemux ! h264parse ! avdec_h264 max-threads=2 output-corrupt=false !                   videoscale method=0 ! video/x-raw,width=640,height=512 ! videoconvert ! videorate max-rate=25 ! video/x-raw,framerate=25/1  !  videoconvert             ! capsfilter ! video/x-raw,format=RGBA ! videoscale ! video/x-raw,width=640,height=512 ! tee name=t ! queue ! videorate ! video/x-raw,framerate=25/1  !  identity name=extraction_utility ! videoconvert ! autovideosink sync=false t.             !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50 name=delay_queue !  videoconvert ! autovideosink sync=true

```

with mux:
```
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 ! tee name=t !  queue name=delay_queue !  autovideosink sync=true  t. ! autovideosink sync=false
```


Rami`s games
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 ! tee name=t !  queue name=delay_queue2 !  autovideosink sync=true  t. ! nvvideoconvert ! queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50 name=delay_queue !  videoconvert ! autovideosink sync=false

with mux:
```
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 !   queue  name=delay_queue !  autovideosink sync=true  

```
GST_DEBUG=3 gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvideoconvert !   nvqueue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50 name=delay_queue  ! 'video/x-raw(memory:NVMM)' ! autovideosink


gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvideoconvert !   queue   name=delay_queue  ! autovideosink




with mux:
```
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 ! tee name=t !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA !  queue max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50 name=delay_queue !  nvvideoconvert  ! videoconvert !  autovideosink sync=true  t. ! autovideosink sync=false
```



WORKS !
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50   name=delay_queue  ! autovideosink sync=true   


WORKS WITH TEE
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder num-extra-surfaces=16 ! queue ! tee name=t ! queue ! autovideosink t. ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50   name=delay_queue  ! autovideosink sync=false

gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream-7.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! avdec_h264 ! nvvideoconvert output-buffers=100 ! queue ! tee name=t ! queue ! autovideosink t. ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50   name=delay_queue  ! autovideosink sync=false


rtsp
```
gst-launch-1.0 rtspsrc location=rtsp://192.168.31.4:8555/video1   ! rtph264depay  !  nvv4l2decoder ! queue  ! m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50   name=delay_queue  ! autovideosink sync=false
```


Based on the pipeline we use
```
GST_DEBUG=3 gst-launch-1.0 rtspsrc location=rtsp://192.168.1.6:8555/video1  drop-on-latency=true protocols=tcp latency=2000 ! rtph264depay  !  nvv4l2decoder ! queue  ! m.sink_0 nvstreammux batch-size=30 live-source=true  width=1280 height=720 name=m batched-push-timeout=33000 ! nvvideoconvert ! capsfilter ! "video/x-raw(memory:NVMM),format=RGBA"  !  nvstreamdemux name=d d.src_0 !  queue ! nvvideoconvert ! nvvideoconvert ! nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=10 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=8   name=delay_queue  ! nvvideoconvert ! nvdsosd process-mode=0 ! nvvideoconvert ! nvh264enc ! decodebin !  autovideosink sync=false
```


future pipeline
```
gst-launch-1.0 rtspsrc location=rtsp://192.168.1.6:8555/video1 drop-on-latency=true latency=400 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! tee name=t ! queue ! autovideosink sync=false t. ! videorate ! "video/x-raw,framerate=25/1" !  nvvideoconvert ! m.sink_0 nvstreammux batch-size=30 live-source=false  width=1280 height=720 name=m batched-push-timeout=40000 frame-duration=-1 attach-sys-ts=true ! nvvideoconvert ! capsfilter ! "video/x-raw(memory:NVMM),format=RGBA"  !  nvstreamdemux name=d d.src_0 !  queue !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50   name=delay_queue  ! autovideosink sync=false
```

```
gst-launch-1.0 rtspsrc location=rtsp://10.11.167.6:8555/video1   ! rtph264depay  !  avdec_h264 ! queue  ! nvvideoconvert ! videorate ! "video/x-raw(memory:NVMM),format=RGBA,framerate=25/1" ! queue !   m.sink_0 nvstreammux batch-size=1 width=1280 height=720 name=m ! nvstreamdemux name=d d.src_0 !  nvvideoconvert ! videoconvert ! capsfilter ! video/x-raw,format=RGBA  !  queue  max-size-buffers=60 max-size-time=2000000000 max-size-bytes= 440401920 min-threshold-time=2000000000 min-threshold-buffers=50   name=delay_queue  ! autovideosink sync=false
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