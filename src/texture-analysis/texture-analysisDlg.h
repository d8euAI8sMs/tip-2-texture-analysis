// texture-analysisDlg.h : header file
//

#pragma once

#include <util/common/gui/SimulationDialog.h>
#include <util/common/gui/PlotControl.h>

#include "model.h"
#include "afxwin.h"

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

    CPlotControl m_plotCtrl;
    model::model_data m_data;
    model::segmentation_helper::autoconfig m_cfg;

    // Generated message map functions
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnBnClickedButton1();
    afx_msg void OnBnClickedButton2();
    CButton m_selectedImageCtrl[4];
    CButton m_features[20];
    CButton m_rolling;
    afx_msg void OnBnClickedRadio2();
};
