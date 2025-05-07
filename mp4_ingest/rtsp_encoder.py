"""
This module will provide encoder functionality
"""

from base64 import decode
from gc import get_stats
from os import stat
from queue import Empty, Full
import sys
import itertools
import numpy as np
import logging
import threading
import multiprocessing as mp
import gi
from pprint import pprint
from logging.handlers import RotatingFileHandler

gi.require_version('Gst', '1.0') 
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GObject, GLib
import cv2
import subprocess
import time

import pickle
import socket
import queue
import os



GObject.threads_init()
Gst.init(None)

def on_debug(category, level, dfile, dfctn, dline, source, message, user_data):
    if source:
        print('Debug {} {}'.format(
            Gst.DebugLevel.get_name(level), message.get()))
    else:
        print('Debug {}: {}'.format(
            Gst.DebugLevel.get_name(level), message.get()))

def set_gst_logger(enable=False, verbosity=None):
    result = True
    try:
        if enable:
            Gst.debug_set_active(True)
            if verbosity and isinstance(verbosity, int) and (verbosity in range(0,7)):
                Gst.debug_set_default_threshold(verbosity)
            else:
                result = False
                logging.error("GST verbosity allowed to be in 0-6")

            Gst.debug_add_log_function(on_debug, None)
            Gst.debug_remove_log_function(Gst.debug_log_default)
        else:
            Gst.debug_set_active(False)

    except Exception as e:
        result = False
    
    return result





class TCPConnectionThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.sock = None

    def connect(self):
        while not self.stop_event.is_set():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                return True
            except Exception as e:
                print("From DS-7 Triton inference server-Error connecting:", e)
                time.sleep(1)  # Retry after 1 second

        return False

    def run(self):
        while not self.stop_event.is_set():
            if not self.sock:
                if not self.connect():
                    # If unable to connect, wait before retrying
                    time.sleep(1)
                    continue

            try:
                while not self.stop_event.is_set():
                    if not self.queue.empty():
                        data = self.queue.get()
                        # Pickle the data
                        serialized_data = pickle.dumps(data)
                        # Calculate the length of the data
                        data_length = len(serialized_data)
                        # Convert data length to bytes (4 bytes for an integer)
                        length_bytes = data_length.to_bytes(4, byteorder='big')
                        # Send the length of the data
                        self.sock.sendall(length_bytes)
                        # Send the data
                        self.sock.sendall(serialized_data)
                    else:
                        time.sleep(0.1)  # Sleep briefly to avoid busy waiting
            except Exception as e:
                print("Error sending data:", e)
                self.sock.close()
                self.sock = None

    def stop(self):
        self.stop_event.set()
        if self.sock:
            self.sock.close()

    def send_message(self, obj):
        try:
            self.queue.put(obj,block=False)
        except Exception as e:
            pass
            # print(e)


# if not Gst.debug_is_active():
#     Gst.debug_set_active(True)
#     level = Gst.debug_get_default_threshold()
#     Gst.debug_set_default_threshold(Gst.DebugLevel.INFO)
#     if level < Gst.DebugLevel.ERROR:
#         Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
#     Gst.debug_add_log_function(on_debug, None)
#     Gst.debug_remove_log_function(Gst.debug_log_default)


# logging.basicConfig(
# level=logging.INFO,
# format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
# handlers=[RotatingFileHandler('GPU_enabled_frame_encode.log',maxBytes=1000000,backupCount=20)])

DEFAULT_BITRATE = 5000

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, fps, img_shape, cols, verbosity=1, cap=None, speed_preset='medium', properties={}):
        super(SensorFactory, self).__init__(**properties)
        logging.info("sensory factory")
        self.rtsp_media = None
        self.height = int(img_shape[0])
        self.width = int(img_shape[1] * cols)
        self.number_frames = 0
        self.stream_timestamp = 0.0
        self.timestamp = time.time()
        self.dt = 0.0
        self.streamed_frames = 0
        self.verbosity = verbosity
        self.fps = int(fps)
        self.cap = cap
        self.appsrc = None
        # duration of a frame in nanoseconds nvh264enc x264enc
        self.INIT_TIME=time.time()
        self.SentFrames_N=0
        self.duration = 1.0 / fps * Gst.SECOND
        key_int_max = ' key-int-max={} '.format(fps)
        caps_str = 'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 '.format(self.width,
                                                                                           self.height,
                                                                                           fps)
        bitrate = 10000 if self.width >= 3840 else DEFAULT_BITRATE
        self.launch_string = 'appsrc name=source is-live=true block=true do-timestamp=true \
            format=GST_FORMAT_TIME ' + caps_str + \
                             ' ! queue' \
                             ' ! videoconvert' \
                             ' ! video/x-raw,format=I420' \
                             ' ! x264enc bitrate={} ' \
                             ' ! rtph264pay config-interval=1 pt=96 name=pay0' \
                             ''.format(str(bitrate))
        # self.launch_string = 'appsrc name=source is-live=true block=true do-timestamp=true \
        #     format=GST_FORMAT_TIME ' + caps_str + \
        #                      ' ! queue' \
        #                      ' ! videoconvert' \
        #                      ' ! video/x-raw,format=I420' \
        #                      ' ! x264enc' \
        #                      ' ! rtph264pay config-interval=1 pt=96 name=pay0'



    def set_cap(self, cap):
        self.cap = cap

    def on_need_data(self, src, length):
        # this method executes when client requests data
        # logging.info("this method executes when client requests data")
        Needed_Frames_tobesent=int(self.fps*(time.time()-self.INIT_TIME))
        while self.SentFrames_N>=Needed_Frames_tobesent:
            time.sleep(0.003)
            Needed_Frames_tobesent = int(self.fps * (time.time() - self.INIT_TIME))
            #print([self.SentFrames_N,Needed_Frames_tobesent])
        # print([self.cap.ID,    self.SentFrames_N,time.ctime(),time.time()])

        self.SentFrames_N=self.SentFrames_N+1
        frame = self.cap.get_canvas()
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        data = frame.tostring()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = self.appsrc.emit('push-buffer', buf)

        if retval != Gst.FlowReturn.OK:
            logging.info("[INFO]: retval not OK: {}".format(retval))
        if retval == Gst.FlowReturn.FLUSHING:
            logging.info('Offline')

        #  the following code has to be in relation with get_canvas from cap
        # if self.verbosity > 0:  
        #     logging.info("[INFO]: Unable to read frame from cap.")

    def do_create_element(self, url):
        if self.verbosity > 0:
            request_uri = url.get_request_uri()
            logging.info('[INFO]: stream request on {}'.format(request_uri))
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.rtsp_media = rtsp_media
        rtsp_media.set_reusable(True)
        self.number_frames = 0
        self.appsrc = rtsp_media.get_element().get_child_by_name('source')
        # executes when client requests data
        self.appsrc.connect('need-data', self.on_need_data)

    def get_rtsp_media(self):
        if self.rtsp_media:
            return self.rtsp_media


class RTSP_utility_server(GstRtspServer.RTSPServer):
    def __init__(self, fps_list=[], suffix='test', rtp_port=8554,
                 ip='12.0.0.0', caps=(None,), Sizes=[[1080, 1920]],
                 speed_preset='medium', verbosity=1, Indexes=[]):
        GObject.threads_init()
        Gst.init(None)
        super(RTSP_utility_server, self).__init__(**{})
        self.verbosity = verbosity
        self.rtp_port = "{}".format(rtp_port)
        if int(self.rtp_port) < 1024 and self.verbosity > 0:
            logging.info(
                '[INFO]: Note, admin privileges are required because port number < 1024.')
        self.set_service(self.rtp_port)
        self.speed_preset = speed_preset
        self.caps = caps
        self.factory = [None] * len(self.caps)
        self.suffix = suffix
        self.fps_list = fps_list
        self.Sizes = Sizes
        self.Indexes = Indexes
        self.attach(None)
        self.ip = self.get_ip()
        self.media_path_list = [None] * len(self.caps)
        self.clients_list = []

        if len(self.suffix):
            self.full_suffix = '/' + self.suffix.lstrip('/')
        else:
            self.full_suffix = ''

        self.connect("client-connected", self.client_connected)
        logging.warning(
            '[INFO]: streaming on:\n\trtsp://{}:{}/{}1'.format(self.ip, self.rtp_port, self.suffix))




        self.context = GLib.MainContext()
        self.context_id = self.attach(self.context)
        self.loop = GObject.MainLoop()

        self.status_thread = threading.Thread(target=self.loop.run)
        self.status_thread.daemon = True
        self.status_thread.start()




    def create_media_factories(self):
        mount_points = self.get_mount_points()
        media_path_list = []
        for i, cap in enumerate(self.caps):
            img_shape = self.Sizes[i]
            fps = self.fps_list[i]
            if len(self.Indexes) == 0:
                N_Index = str(i + 1)
            else:
                N_Index = str(self.Indexes[i])
            factory = SensorFactory(fps=fps, img_shape=img_shape, speed_preset=self.speed_preset,
                                    cols=1, verbosity=self.verbosity, cap=cap)
            factory.set_shared(True)
            factory.set_stop_on_disconnect(True)

            logging.info('inside media_factories Stream on ' +
                         self.full_suffix + N_Index)
            logging.info('inside media_factories Stream on ' +
                         self.full_suffix + N_Index)

            mount_points.add_factory(self.full_suffix + N_Index, factory)
            self.factory[i] = factory
            media_path_list.append(self.full_suffix + N_Index)
            self.media_path_list = media_path_list


    def destroy_media_factories(self):
        session_pool = self.get_session_pool()
        logging.info("Number of sessions are :" +
                     str(session_pool.get_n_sessions()))
        sessions_list = session_pool.filter()
        for session in sessions_list:
            for path in self.get_paths():
                media_matched, _ = session.get_media(path)
                if media_matched:
                    rtsp_media = media_matched.get_media()
                    rtsp_media.set_eos_shutdown(True)
                    rtsp_media.unprepare()
                    logging.debug("media removed for path "+path)
        number_of_disconnects = session_pool.cleanup()
        if number_of_disconnects > 0:
            logging.info("number of disconnects:"+str(number_of_disconnects))


    def disconnect_all(self, a, b):
        return GstRtspServer.RTSPFilterResult.REMOVE
        
    def destroy_media_factories_by_path(self,path_to_remove="/video1"):
        session_pool = self.get_session_pool()
        logging.info("Number of sessions are :" +
                     str(session_pool.get_n_sessions()))
        sessions_list = session_pool.filter()
        for session in sessions_list:
            for path in self.get_paths():
                media_matched, _ = session.get_media(path)
                if media_matched and path==path_to_remove:
                    rtsp_media = media_matched.get_media()
                    rtsp_media.set_eos_shutdown(True)
                    rtsp_media.unprepare()
                    logging.debug("media removed for path "+path)
        number_of_disconnects = session_pool.cleanup()
        if number_of_disconnects > 0:
            logging.info("number of disconnects:"+str(number_of_disconnects))

    def client_connected(self, gst_server_obj, rtsp_client_obj):
        logging.info('[INFO]: Client has connected')
        self.create_media_factories()
        self.clients_list.append(rtsp_client_obj)
        if self.verbosity > 0:
            logging.info('[INFO]: Client has connected')


    def stop(self):
        self.loop.quit()


    def stop_all(self):
        self.destroy_media_factories()

    def stop_by_index(self,path):
        self.destroy_media_factories_by_path(path)

    def get_paths(self):
        return self.media_path_list

    def get_status(self):

        mount_points = self.get_mount_points()
        session_pool = self.get_session_pool()
        # logging.info("Number of sessions are :" +
        #              str(session_pool.get_n_sessions()))
        number_of_disconnects = session_pool.cleanup()
        if number_of_disconnects > 0:
            logging.info("number of disconnects:"+str(number_of_disconnects))          
        for path in self.get_paths():
            sessions_list = session_pool.filter()
            for session in sessions_list:
                session.set_timeout(1)
                media_matched, _ = session.get_media(path)
                if media_matched:
                    rtsp_media = media_matched.get_media()
                    status = rtsp_media.get_status() #<enum GST_RTSP_MEDIA_STATUS_PREPARED of type GstRtspServer.RTSPMediaStatus>
                    
                    if "GST_RTSP_MEDIA_STATUS_UNPREPARING" in str(status):
                        logging.critical(f'{path} - Improper Teardown !!')
                        print("Improper Teardown")



    def get_current_encoders_details(self):
        enc_details = dict()
        for i,cap in enumerate(self.caps):
            enc_details[i]={"path":self.media_path_list[i],"empty_screen_filler":cap,"factory":self.factory[i]}
        return enc_details

    
    def shutdown(self):
        self.loop.quit()
        logging.info("Encoder main loop ended")

    @staticmethod
    def get_ip():
        return subprocess.check_output("hostname -I", shell=True).decode('utf-8').split(' ')[0]


class encoders():
    """
    This is the containter for multiple encoders
    """
    _default_resolutions = (1080, 1920)
    _default_fps = 30

    def __init__(self,empty_screen_fillers=[],head=[],resolutions=[],suffix="video",fps_list=[],rtp_port=8554,Indexes=[]) -> None:
        
        self.encoder_list = []
        self.caps = []
        self.Sizes = []
        self.fps_list = []

        if len(Indexes) != len(empty_screen_fillers):
            logging.error(f'Number of Indexes and Cap objects does not match, Please fix it')
            sys.exit(1)
        for i, capobj in enumerate(empty_screen_fillers):
            self.caps.append(capobj)

            if i < len(resolutions):
                self.Sizes.append(resolutions[i])
            else:
                logging.warning(f'Number of resolutions and Cap objects does not match, default values will be applied for missing resoltions accordingly')
                self.Sizes.append(encoders._default_resolutions)
                            
            if i < len(fps_list):
                self.fps_list.append(fps_list[i])
            else: 
                logging.warning(f'Number of fps and Cap objects does not match, default values will be applied for missing fps accordingly')
                self.fps_list.append(encoders._default_fps)

            
                
            


            
                

        self.rtp_port = rtp_port
        self.enc_obj = RTSP_utility_server(fps_list=self.fps_list, Sizes=self.Sizes, caps=empty_screen_fillers, suffix=suffix,
                                           verbosity=1, rtp_port=self.rtp_port,Indexes=Indexes)

    def get_encoders(self):
        return self.enc_obj.get_current_encoders_details()


    def stop_all(self):
        logging.warning('Request to stop all encoders initated..')
        self.enc_obj.stop_all()
        logging.info('All encoders stopped.')
        

    def get_status(self):
        self.enc_obj.get_status()


    def shutdown(self):
        logging.warning('Requested to shutdown the Encoder')
        self.enc_obj.stop_all()
        logging.info('Stoppped all the encoder resources')
        self.enc_obj.shutdown()
        logging.info('Encoder has stopped.')



import numpy as np
# Start the TCP connection thread
host = "127.0.0.1"  # Change to your server's IP
port = 9876         # Change to your server's port

tcp_thread = TCPConnectionThread(host, port)
tcp_thread.start()

class Cap_Gstremer:
    def __init__(self):
        self.k=0
        image=np.zeros((300,300,3)).astype('uint8')
        self.Last=image.copy()
        A = threading.Thread(target=self.image_create)
        A.daemon = True
        A.start()
        self.cap = None


    def image_create(self):
        index = 0
        vids = ['1.mp4']*3
        self.cap = cv2.VideoCapture(vids[index])
        while True:
            ret, image=self.cap.read()
            # image=np.zeros((300,300,3)).astype('uint8')
            cv2.putText(image, f"src_#:{str(self.k)}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 4)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            milliseconds = int((time.time() % 1) * 1000)
            current_time_with_ms = f"{current_time}.{milliseconds:03d}"
            cv2.putText(image, f"from source, time now:{current_time_with_ms}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            tcp_thread.send_message({0:current_time_with_ms})
            self.k=self.k+1
            if ret:
                self.Last=image.copy()
            else:
                cv2.putText(image, str("Restarting"), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                self.cap.release()
                index = index + 1
                if index >2 :
                    index = 0
                self.cap = cv2.VideoCapture(vids[index])
            time.sleep(1/30)


    def isOpened(self):
        return True

    def get_canvas(self):
        A=self.Last
        return A

Cam1=Cap_Gstremer()
kk=1




speed_preset = 'fast'
rtsp_server = RTSP_utility_server(fps_list=[30], Sizes=[[480,640]],caps=[Cam1], suffix='video', verbosity=0, rtp_port=8555, ip='10.5.1.130')

while True:
    print("sleeping 40 secs")
    time.sleep(40)

# kk=1
# #cv2.namedWindow('A', cv2.WINDOW_NORMAL)
# #cv2.resizeWindow('A', 640, 360)
# while True:
#     time.sleep(2)

    #image=Cam1.Last.copy()
    #cv2.imshow('A',image)
    #k=cv2.waitKey(33)
    #if k==27:
    #    break
#cv2.destroyAllWindows()
#rtsp_server.destroy_media_factories()