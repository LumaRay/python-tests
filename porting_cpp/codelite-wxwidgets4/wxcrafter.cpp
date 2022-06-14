//////////////////////////////////////////////////////////////////////
// This file was auto-generated by codelite's wxCrafter Plugin
// wxCrafter project file: wxcrafter.wxcp
// Do not modify this file by hand!
//////////////////////////////////////////////////////////////////////

#include "wxcrafter.h"

// Declare the bitmap loading function
extern void wxC9ED9InitBitmapResources();

static bool bBitmapLoaded = false;

MainFrameBaseClass::MainFrameBaseClass(wxWindow *parent, wxWindowID id,
                                       const wxString &title,
                                       const wxPoint &pos, const wxSize &size,
                                       long style)
    : wxFrame(parent, id, title, pos, size, style) {
  if (!bBitmapLoaded) {
    // We need to initialise the default bitmap handler
    wxXmlResource::Get()->AddHandler(new wxBitmapXmlHandler);
    wxC9ED9InitBitmapResources();
    bBitmapLoaded = true;
  }

  wxBoxSizer *boxSizer1 = new wxBoxSizer(wxHORIZONTAL);
  this->SetSizer(boxSizer1);

  m_splitter15 = new wxSplitterWindow(this, wxID_ANY, wxDefaultPosition,
                                      wxDLG_UNIT(this, wxSize(-1, -1)),
                                      wxSP_LIVE_UPDATE | wxSP_3D);
  m_splitter15->SetSashGravity(1);
  m_splitter15->SetMinimumPaneSize(20);

  boxSizer1->Add(m_splitter15, 1, wxALL | wxEXPAND, WXC_FROM_DIP(5));

  m_splitterPage19 = new wxPanel(m_splitter15, wxID_ANY, wxDefaultPosition,
                                 wxDLG_UNIT(m_splitter15, wxSize(-1, -1)),
                                 wxFULL_REPAINT_ON_RESIZE | wxTAB_TRAVERSAL);

  wxBoxSizer *boxSizer31 = new wxBoxSizer(wxVERTICAL);
  m_splitterPage19->SetSizer(boxSizer31);

  int *m_glCanvas33Attr = NULL;
  m_glCanvas33 = new wxGLCanvas(
      m_splitterPage19, wxID_ANY, m_glCanvas33Attr, wxDefaultPosition,
      wxDLG_UNIT(m_splitterPage19, wxSize(50, 50)), 0);
  wxDELETEA(m_glCanvas33Attr);

  boxSizer31->Add(m_glCanvas33, 1, wxEXPAND, WXC_FROM_DIP(5));

  m_splitterPage23 = new wxPanel(m_splitter15, wxID_ANY, wxDefaultPosition,
                                 wxDLG_UNIT(m_splitter15, wxSize(-1, -1)),
                                 wxFULL_REPAINT_ON_RESIZE | wxTAB_TRAVERSAL);
  m_splitter15->SplitVertically(m_splitterPage19, m_splitterPage23, 0);

  wxBoxSizer *boxSizer25 = new wxBoxSizer(wxHORIZONTAL);
  m_splitterPage23->SetSizer(boxSizer25);

  wxArrayString m_pgMgr27Arr;
  wxUnusedVar(m_pgMgr27Arr);
  wxArrayInt m_pgMgr27IntArr;
  wxUnusedVar(m_pgMgr27IntArr);
  m_pgMgr27 = new wxPropertyGridManager(
      m_splitterPage23, wxID_ANY, wxDefaultPosition,
      wxDLG_UNIT(m_splitterPage23, wxSize(-1, -1)),
      wxPG_DESCRIPTION | wxPG_TOOLBAR | wxPG_TOOLTIPS |
          wxPG_SPLITTER_AUTO_CENTER | wxPG_BOLD_MODIFIED);

  boxSizer25->Add(m_pgMgr27, 1, wxEXPAND, WXC_FROM_DIP(5));

  m_pgProp29 = m_pgMgr27->Append(
      new wxStringProperty(_("My Label4"), wxPG_LABEL, wxT("")));
  m_pgProp29->SetHelpString(wxT(""));

  m_pgProp35 = m_pgMgr27->Append(new wxPropertyCategory(_("My Label6")));
  m_pgProp35->SetHelpString(wxT(""));

  m_pgProp37 = m_pgMgr27->AppendIn(
      m_pgProp35, new wxStringProperty(_("My Label8"), wxPG_LABEL, wxT("")));
  m_pgProp37->SetHelpString(wxT(""));

  m_pgProp39 =
      m_pgMgr27->Append(new wxBoolProperty(_("My Label10"), wxPG_LABEL, 1));
  m_pgProp39->SetHelpString(wxT(""));
  m_pgProp39->SetEditor(wxT("CheckBox"));

  m_pgProp41 =
      m_pgMgr27->Append(new wxIntProperty(_("My Label12"), wxPG_LABEL, 0));
  m_pgProp41->SetHelpString(wxT(""));
  m_pgProp41->SetEditor(wxT("SpinCtrl"));

  SetName(wxT("MainFrameBaseClass"));
  SetSize(wxDLG_UNIT(this, wxSize(500, 300)));
  if (GetSizer()) {
    GetSizer()->Fit(this);
  }
  if (GetParent()) {
    CentreOnParent();
  } else {
    CentreOnScreen();
  }
#if wxVERSION_NUMBER >= 2900
  if (!wxPersistenceManager::Get().Find(this)) {
    wxPersistenceManager::Get().RegisterAndRestore(this);
  } else {
    wxPersistenceManager::Get().Restore(this);
  }
#endif
  // Connect events
  m_glCanvas33->Connect(wxEVT_LEFT_UP,
                        wxMouseEventHandler(MainFrameBaseClass::onGlLeftUp),
                        NULL, this);
}

MainFrameBaseClass::~MainFrameBaseClass() {
  m_glCanvas33->Disconnect(wxEVT_LEFT_UP,
                           wxMouseEventHandler(MainFrameBaseClass::onGlLeftUp),
                           NULL, this);
}