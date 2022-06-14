#include <wx/app.h>
#include <wx/event.h>
#include "MainFrame.h"
#include <wx/image.h>

//  /home/thermalview/Phoenix/build/wxbld/gtk3/lib/
//  libwx_gtk3u_propgrid-3.1.so;libwx_gtk3u_gl-3.1.so
//  create (as root) a new file in /etc/ld.so.conf.d/ containing, the new path. For example:
//  sudo touch /etc/ld.so.conf.d/codelite-wx.conf
//  sudo echo "/home/thermalview/Phoenix/build/wxbld/gtk3/lib/" >> /etc/ld.so.conf.d/codelite-wx.conf
//  sudo ldconfig
//  ldd exec_name
//  wx-config --list
// /usr/bin/wx-config is a symbolic link to one of the files under /usr/lib/wx/confi
// g. The file that it links to is the default build and the others are the alternatives listed by wx-config --list.
// ln -s /usr/lib/wx/config/gtk2-unicode-release-2.8 /usr/bin/wx-config
//  ln -s /home/thermalview/Phoenix/build/wxbld/gtk3/wx-config /usr/bin/wx-config
// old = ln -s /etc/alternatives/wx-config /usr/bin/wx-config
// /home/thermalview/Phoenix/build/wxbld/gtk3/lib/
// libwx_gtk3u_core-3.1.so;libwx_gtk3u_adv-3.1.so;libwx_gtk3u_xrc-3.1.so;libwx_gtk3u_propgrid-3.1.so;libwx_gtk3u_gl-3.1.so
// sudo echo "/home/thermalview/.local/lib/" >> /etc/ld.so.conf.d/opencv.conf
// sudo ldconfig
// LIBS += -ltorch -lnvrtc -lc10
// LIBS += -lcudnn -lcuda -lcudart -lcublas -lcurand -lcusolver
// ;-D_GLIBCXX_USE_CXX11_ABI=0
// https://kezunlin.me/post/54e7a3d8/
// /home/thermalview/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include/
// /home/thermalview/.local/lib/python3.6/site-packages/torch/include/
// /home/thermalview/.local/lib/
// /home/thermalview/.local/lib/python3.6/site-packages/torch/lib/
// .;/home/thermalview/.local/include/opencv4/;/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include/
// /home/thermalview/Phoenix/build/wxbld/gtk3/lib/;/usr/local/cuda-10.1/targets/x86_64-linux/lib/
// libwx_baseu-3.1.so;libwx_gtk3u_stc-3.1.so;libwx_baseu_net-3.1.so;libwx_gtk3u_media-3.1.so;libwx_gtk3u_aui-3.1.so;libwx_gtk3u_ribbon-3.1.so;libwx_gtk3u_html-3.1.so;libwx_gtk3u_richtext-3.1.so;libwx_baseu_xml-3.1.so;libwx_gtk3u_core-3.1.so;libwx_gtk3u_adv-3.1.so;libwx_gtk3u_xrc-3.1.so;libwx_gtk3u_propgrid-3.1.so;libwx_gtk3u_gl-3.1.so;libopencv_core.so;libopencv_imgproc.so;libopencv_imgcodecs.so;libopencv_highgui.so;libtorch.so;libtorch_global_deps.so;libshm.so;libnvrtc.so;libcudnn.so;libcuda.so;libcudart.so;libcublas.so;libcurand.so;libcusolver.so;libnvrtc-builtins.so;libnvToolsExt.so;libcaffe2_observers.so;libcaffe2_nvrtc.so;libcaffe2_module_test_dynamic.so;libcaffe2_detectron_ops_gpu.so;libc10.so;libc10_cuda.so
//  sudo ldconfig
//  sudo ldconfig
//  sudo ldconfig

// Define the MainApp
class MainApp : public wxApp
{
public:
    MainApp() {}
    virtual ~MainApp() {}

    virtual bool OnInit() {
        // Add the common image handlers
        wxImage::AddHandler( new wxPNGHandler );
        wxImage::AddHandler( new wxJPEGHandler );

        MainFrame *mainFrame = new MainFrame(NULL);
        SetTopWindow(mainFrame);
        return GetTopWindow()->Show();
    }
};

DECLARE_APP(MainApp)
IMPLEMENT_APP(MainApp)
