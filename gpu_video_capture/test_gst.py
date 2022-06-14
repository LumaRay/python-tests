import threading
import time
# import thread
# import gobject
# import pygst
# pygst.require("0.10")
# import gst
import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject as gobject

Gst.init(None)
# https://brettviren.github.io/pygst-tutorial-org/pygst-tutorial.html

class VideoPlayer:
    """
    Simple Video player that just 'plays' a valid input Video file.
    """
    def __init__(self):
        self.use_parse_launch = False
        self.decodebin = None
        self.inFileLocation = "/home/thermalview/Downloads/X2Convert.com what_language_do_you_find_most_attractive_street_interviews_-8052641072221435869.mp4"

        self.constructPipeline()
        self.is_playing = False
        self.connectSignals()

        th = threading.Thread(target=self.play, args=())
        th.start()

    def constructPipeline(self):
        """
        Add and link elements in a GStreamer pipeline.
        """
        # Create the pipeline instance
        self.player = Gst.Pipeline()

        # Define pipeline elements
        self.filesrc = Gst.ElementFactory.make("filesrc")
        self.filesrc.set_property("location", self.inFileLocation)
        self.decodebin = Gst.ElementFactory.make("decodebin")

        # audioconvert for audio processing pipeline
        self.audioconvert = Gst.ElementFactory.make("audioconvert")

        # Autoconvert element for video processing
        self.autoconvert = Gst.ElementFactory.make("autoconvert")

        self.audiosink = Gst.ElementFactory.make("autoaudiosink")

        self.videosink = Gst.ElementFactory.make("autovideosink")

        # As a precaution add videio capability filter
        # in the video processing pipeline.
        videocap = Gst.Caps("video/x-raw-yuv")
        self.filter = Gst.ElementFactory.make("capsfilter")
        self.filter.set_property("caps", videocap)
        # Converts the video from one colorspace to another
        # self.colorSpace = Gst.ElementFactory.make("ffmpegcolorspace")
        self.colorSpace = Gst.ElementFactory.make("videoconvert")

        self.queue1 = Gst.ElementFactory.make("queue")
        self.queue2 = Gst.ElementFactory.make("queue")

        # Add elements to the pipeline
        self.player.add(self.filesrc,
                        self.decodebin,
                        self.autoconvert,
                        self.audioconvert,
                        self.queue1,
                        self.queue2,
                        self.filter,
                        self.colorSpace,
                        self.audiosink,
                        self.videosink)

        # Link elements in the pipeline.
        self.filesrc.link(self.decodebin)
        self.queue1.link(self.autoconvert)
        self.autoconvert.link(self.filter)
        self.filter.link(self.colorSpace)
        self.colorSpace.link(self.videosink)
        self.queue2.link(self.audioconvert)
        self.audioconvert.link(self.audiosink)

    def connectSignals(self):
        """
        Connects signals with the methods.
        """
        # Capture the messages put on the bus.
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.message_handler)

        # Connect the decodebin signal
        if not self.decodebin is None:
            self.decodebin.connect("pad_added", self.decodebin_pad_added)

    def decodebin_pad_added(self, decodebin, pad):
        """
        Manually link the decodebin pad with a compatible pad on
        queue elements, when the decodebin "pad-added" signal
        is generated.
        """
        compatible_pad = None
        caps = pad.get_caps()
        name = caps[0].get_name()
        print("\n cap name is = ", name)
        if name[:5] == 'video':
            compatible_pad = self.queue1.get_compatible_pad(pad, caps)
        elif name[:5] == 'audio':
            compatible_pad = self.queue2.get_compatible_pad(pad, caps)

        if compatible_pad:
            pad.link(compatible_pad)

    def play(self):
        """
        Play the media file
        """
        self.is_playing = True
        self.player.set_state(Gst.State.PLAYING)
        while self.is_playing:
            time.sleep(1)
        evt_loop.quit()

    def message_handler(self, bus, message):
        """
        Capture the messages on the bus and
        set the appropriate flag.
        """
        msgType = message.type
        '''print(msgType)
        if msgType == Gst.MessageType.ERROR:
            self.player.set_state(Gst.State.NULL)
            self.is_playing = False
            print("\n Unable to play Video. Error: ", message.parse_error())
        elif msgType == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.is_playing = False'''

# Run the program
player = VideoPlayer()
# thread.start_new_thread(player.play, ())
gobject.threads_init()
evt_loop = gobject.MainLoop()
evt_loop.run()

