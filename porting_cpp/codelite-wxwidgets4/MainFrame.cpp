#include "MainFrame.h"
#include <wx/aboutdlg.h>
#include "test_opencv.h"
#include "test_torch.h"
#include "test_opengl.h"
//#include <GL/glut.h>

MainFrame::MainFrame(wxWindow* parent)
    : MainFrameBaseClass(parent)
{
}

MainFrame::~MainFrame()
{
}

void MainFrame::OnExit(wxCommandEvent& event)
{
    wxUnusedVar(event);
    Close();
}

void MainFrame::OnAbout(wxCommandEvent& event)
{
    wxUnusedVar(event);
    wxAboutDialogInfo info;
    info.SetCopyright(_("My MainFrame"));
    info.SetLicence(_("GPL v2 or later"));
    info.SetDescription(_("Short description goes here"));
    ::wxAboutBox(info);
}
void MainFrame::onFrameSize(wxSizeEvent& event)
{
}
void MainFrame::onGlLeftUp(wxMouseEvent& event)
{
    tst_torch();
    tst_opencv();
    //this->m_glCanvas33->
}
