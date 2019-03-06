// texture-analysisDlg.h : header file
//

#pragma once

#include <util/common/gui/SimulationDialog.h>

// CTextureAnalysisDlg dialog
class CTextureAnalysisDlg : public CSimulationDialog
{
// Construction
public:
    CTextureAnalysisDlg(CWnd* pParent = NULL);    // standard constructor

// Dialog Data
    enum { IDD = IDD_TEXTUREANALYSIS_DIALOG };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
    HICON m_hIcon;

    // Generated message map functions
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    DECLARE_MESSAGE_MAP()
};
