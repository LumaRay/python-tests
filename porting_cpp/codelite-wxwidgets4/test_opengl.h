#include <wx/wx.h>
#include <wx/glcanvas.h>

//#include <GL/glu.h>
//#include <GL/gl.h>

#include <GL/glut.h>

class wxGLCanvasSubClass: public wxGLCanvas {
        void Render();
public:
    wxGLCanvasSubClass(wxFrame* parent);
    void Paintit(wxPaintEvent& event);
protected:
    DECLARE_EVENT_TABLE()
};