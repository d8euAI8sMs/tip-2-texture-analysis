// texture-analysisDlg.cpp : implementation file
//

#include "stdafx.h"
#include "texture-analysis.h"
#include "texture-analysisDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CTextureAnalysisDlg dialog

CTextureAnalysisDlg::CTextureAnalysisDlg(CWnd* pParent /*=NULL*/)
    : CSimulationDialog(CTextureAnalysisDlg::IDD, pParent)
    , m_sTrainingSets(_T("sky,clouds"))
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
    m_data.params = model::make_default_parameters();
    m_cfg = model::segmentation_helper::make_default_config();
}

void CTextureAnalysisDlg::DoDataExchange(CDataExchange* pDX)
{
    CSimulationDialog::DoDataExchange(pDX);
    DDX_Control(pDX, IDC_PIC, m_plotCtrl);
    DDX_Control(pDX, IDC_RADIO2, m_selectedImageCtrl[0]);
    DDX_Control(pDX, IDC_RADIO3, m_selectedImageCtrl[1]);
    DDX_Control(pDX, IDC_RADIO4, m_selectedImageCtrl[2]);
    DDX_Control(pDX, IDC_RADIO5, m_selectedImageCtrl[3]);
    DDX_Control(pDX, IDC_CHECK2, m_features[0]);
    DDX_Control(pDX, IDC_CHECK3, m_features[1]);
    DDX_Control(pDX, IDC_CHECK4, m_features[2]);
    DDX_Control(pDX, IDC_CHECK5, m_features[3]);
    DDX_Control(pDX, IDC_CHECK6, m_features[4]);
    DDX_Control(pDX, IDC_CHECK8, m_features[5]);
    DDX_Control(pDX, IDC_CHECK9, m_features[6]);
    DDX_Control(pDX, IDC_CHECK10, m_features[7]);
    DDX_Control(pDX, IDC_CHECK11, m_features[8]);
    DDX_Control(pDX, IDC_CHECK12, m_features[9]);
    DDX_Control(pDX, IDC_CHECK13, m_features[10]);
    DDX_Control(pDX, IDC_CHECK14, m_features[11]);
    DDX_Control(pDX, IDC_CHECK7, m_rolling);
    DDX_Text(pDX, IDC_EDIT1, m_cfg.wnd_size);
    DDX_Text(pDX, IDC_EDIT2, m_cfg.histo_cols);
    DDX_Text(pDX, IDC_EDIT3, m_cfg.cooccurrence_cols);
    DDX_Text(pDX, IDC_EDIT4, m_cfg.gabor_thetas);
    DDX_Text(pDX, IDC_EDIT5, m_cfg.gabor_lambdas);
    DDX_Text(pDX, IDC_EDIT6, m_cfg.kmeans_clusters);
    DDX_Text(pDX, IDC_EDIT7, m_sTrainingSets);
}

BEGIN_MESSAGE_MAP(CTextureAnalysisDlg, CSimulationDialog)
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    ON_BN_CLICKED(IDC_BUTTON1, &CTextureAnalysisDlg::OnBnClickedButton1)
    ON_BN_CLICKED(IDC_BUTTON2, &CTextureAnalysisDlg::OnBnClickedButton2)
    ON_BN_CLICKED(IDC_RADIO2, &CTextureAnalysisDlg::OnBnClickedRadio2)
    ON_BN_CLICKED(IDC_RADIO3, &CTextureAnalysisDlg::OnBnClickedRadio2)
    ON_BN_CLICKED(IDC_RADIO4, &CTextureAnalysisDlg::OnBnClickedRadio2)
    ON_BN_CLICKED(IDC_RADIO5, &CTextureAnalysisDlg::OnBnClickedRadio2)
    ON_BN_CLICKED(IDC_BUTTON3, &CTextureAnalysisDlg::OnBnClickedButton3)
    ON_BN_CLICKED(IDC_BUTTON4, &CTextureAnalysisDlg::OnBnClickedButton4)
END_MESSAGE_MAP()

// CTextureAnalysisDlg message handlers

BOOL CTextureAnalysisDlg::OnInitDialog()
{
    CSimulationDialog::OnInitDialog();

    // Set the icon for this dialog.  The framework does this automatically
    //  when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);            // Set big icon
    SetIcon(m_hIcon, FALSE);        // Set small icon

    // TODO: Add extra initialization here

    m_plotCtrl.plot_layer.with(model::make_bmp_plot(m_data.csource));
    m_plotCtrl.plot_layer.with(model::make_bmp_plot(m_data.cmask));
    m_plotCtrl.plot_layer.with(model::make_bmp_plot(m_data.cresult));
    m_plotCtrl.plot_layer.with(model::make_bmp_plot(m_data.cfeatmap));

    m_plotCtrl.plot_layer.layers[1]->visible = false;
    m_plotCtrl.plot_layer.layers[2]->visible = false;
    m_plotCtrl.plot_layer.layers[3]->visible = false;

    m_features[0].SetCheck(1);
    m_selectedImageCtrl[0].SetCheck(1);

    return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CTextureAnalysisDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else
    {
        CSimulationDialog::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CTextureAnalysisDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}


void CTextureAnalysisDlg::OnBnClickedButton1()
{
    CFileDialog fd(TRUE, TEXT("bmp"));
    if (fd.DoModal() == IDOK)
    {
        std::wstring path(fd.GetPathName().GetBuffer());
        std::string asciipath(path.begin(), path.end());
        m_data.source.from_file(asciipath);
        m_data.source.to_cbitmap(m_data.csource);
        m_plotCtrl.RedrawWindow();
    }
    OnBnClickedButton2();
}


void CTextureAnalysisDlg::UpdateConfig()
{
    UpdateData(TRUE);

    m_cfg.features = 0;
    if (m_features[0].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_mean;
    if (m_features[1].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_std;
    if (m_features[2].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_coords;
    if (m_features[3].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_histo;
    if (m_features[4].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_entropy;
    if (m_features[5].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_cooccurrence;
    if (m_features[6].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_coasm;
    if (m_features[7].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_cocontrast;
    if (m_features[8].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_cohomogenity;
    if (m_features[9].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_cocorrelation;
    if (m_features[10].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_coentropy;
    if (m_features[11].GetCheck()) m_cfg.features |= model::segmentation_helper::feature_gabor;
    
    m_cfg.wnd_roll = (m_rolling.GetCheck() == 1);

    std::wstring wtrainsets(m_sTrainingSets.GetBuffer());
    std::string trainsets(std::begin(wtrainsets), std::end(wtrainsets));

    std::vector < std::string > trainsetlist;

    while (!trainsets.empty()) 
    {
        size_t idx = trainsets.find(',');
        if (idx == std::string::npos)
        {
            if (!model::trim(trainsets).empty()) trainsetlist.push_back(trainsets);
            break;
        }
        else
        {
            std::string trainset = model::trim(trainsets.substr(0, idx));
            if (!trainset.empty()) trainsetlist.push_back(trainset);
            trainsets = trainsets.substr(idx + 1);
        }
    }

    WIN32_FIND_DATA findData;
	HANDLE handle;

    m_trainset.clear();

    for (size_t i = 0; i < trainsetlist.size(); ++i)
    {
        auto path = "trainsets\\" + trainsetlist[i] + "\\*";
        std::wstring wpath(std::begin(path), std::end(path));
        CString cpath = wpath.c_str();
        handle = FindFirstFile(cpath, &findData);
        // skip . and ..
        FindNextFile(handle, &findData);
        FindNextFile(handle, &findData);
        do {
            std::wstring wfile = findData.cFileName;
            std::string file(std::begin(wfile), std::end(wfile));
            file = "trainsets\\" + trainsetlist[i] + "\\" + file;
            m_trainset.emplace_back(i, std::move(file));
		} while (FindNextFile(handle, &findData));
		FindClose(handle);
    }
}


void CTextureAnalysisDlg::OnBnClickedButton2()
{
    UpdateConfig();
    model::segmentation_helper h(m_data.params);
    h.autoprocess(m_data.source, m_data.featmap, m_data.mask, m_data.result, m_cfg);
    m_data.mask.to_cbitmap(m_data.cmask);
    m_data.result.to_cbitmap(m_data.cresult);
    m_data.featmap.to_cbitmap(m_data.cfeatmap);
    m_plotCtrl.RedrawWindow();
}


void CTextureAnalysisDlg::OnBnClickedRadio2()
{
    m_plotCtrl.plot_layer.layers[0]->visible = (m_selectedImageCtrl[0].GetCheck() == 1);
    m_plotCtrl.plot_layer.layers[1]->visible = (m_selectedImageCtrl[1].GetCheck() == 1);
    m_plotCtrl.plot_layer.layers[2]->visible = (m_selectedImageCtrl[2].GetCheck() == 1);
    m_plotCtrl.plot_layer.layers[3]->visible = (m_selectedImageCtrl[3].GetCheck() == 1);
    m_plotCtrl.RedrawWindow();
}


void CTextureAnalysisDlg::OnBnClickedButton3()
{
    UpdateConfig();

    std::vector < std::pair < int, cv::Mat > > samples;

    model::bitmap bmp;

    for each (auto & s in m_trainset)
    {
        samples.emplace_back(s.first, bmp.from_file(s.second).mat);
    }

    m_pKnearest = cv::ml::KNearest::create();
    model::segmentation_helper h(m_data.params);
    h.train_knearest(samples, *m_pKnearest, m_cfg);
}


void CTextureAnalysisDlg::OnBnClickedButton4()
{
    UpdateConfig();
    model::segmentation_helper h(m_data.params);
    h.knearest_process(m_data.source, *m_pKnearest, m_data.featmap, m_data.mask, m_data.result, m_cfg);
    m_data.mask.to_cbitmap(m_data.cmask);
    m_data.result.to_cbitmap(m_data.cresult);
    m_data.featmap.to_cbitmap(m_data.cfeatmap);
    m_plotCtrl.RedrawWindow();
}
