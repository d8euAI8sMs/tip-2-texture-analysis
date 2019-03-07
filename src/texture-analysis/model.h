#pragma once

#include <afxwin.h>

#include <vector>
#include <map>
#include <cstdint>

#include <util/common/math/complex.h>
#include <util/common/plot/plot.h>
#include <util/common/math/fft.h>
#include <util/common/math/vec.h>
#include <util/common/math/raster.h>
#include <util/common/geom/geom.h>

#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif // !M_PI
#ifndef M_E
#define M_E 2.7182818284590452353602874713527
#endif // !M_PI


namespace model
{

    /*****************************************************/
    /*                     params                        */
    /*****************************************************/

    struct parameters
    {
    };

    inline parameters make_default_parameters()
    {
        parameters p =
        {
        };
        return p;
    }

    inline plot::drawable::ptr_t make_bmp_plot(CBitmap & b)
    {
        return plot::custom_drawable::create([&b] (CDC & dc, const plot::viewport & vp)
        {
            if (!b.m_hObject) return;
            CDC memDC; memDC.CreateCompatibleDC(&dc);
            memDC.SelectObject(&b);
            dc.SetStretchBltMode(HALFTONE);
            auto wh = b.GetBitmapDimension();
            dc.StretchBlt(vp.screen.xmin, vp.screen.ymin,
                          vp.screen.width(), vp.screen.height(),
                          &memDC, 0, 0, wh.cx, wh.cy, SRCCOPY);
        });
    }

    /*****************************************************/
    /*                     data                          */
    /*****************************************************/

    struct bitmap
    {
        cv::Mat mat;

        bitmap & to_cbitmap(CBitmap & bmp)
        {
            std::vector < COLORREF > buf(mat.rows * mat.cols);
            for (size_t i = 0; i < mat.rows; ++i)
            for (size_t j = 0; j < mat.cols; ++j)
            {
                if (mat.channels() == 1)
                {
                    float v = mat.at < float > (i, j);
                    BYTE c = (BYTE) (v * 255);
                    buf[mat.cols * i + j] = RGB(c, c, c);
                }
                else
                {
                    auto v = mat.at < cv::Vec3f > (i, j);
                    BYTE c[3] = { (BYTE) (v[0] * 255), (BYTE) (v[1] * 255), (BYTE) (v[2] * 255) };
                    buf[mat.cols * i + j] = RGB(c[0], c[1], c[2]);
                }
            }
            bmp.DeleteObject();
            bmp.CreateBitmap(mat.cols, mat.rows, 1, sizeof(COLORREF) * 8, (LPCVOID) buf.data());
            bmp.SetBitmapDimension(mat.cols, mat.rows);
            return *this;
        }
        bitmap & from_file(const std::string & path)
        {
            auto img = cv::imread(path);
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            mat.create(img.rows, img.cols, CV_32F);
            for (size_t i = 0; i < mat.rows; ++i)
            for (size_t j = 0; j < mat.cols; ++j)
            {
                mat.at < float > (i, j) = img.at < std::uint8_t > (i, j) / 255.0;
            }
            return *this;
        }
    };

    /*****************************************************/
    /*                     algo                          */
    /*****************************************************/

    class segmentation_helper
    {
    public:
        static const size_t feature_global        = 1 << 31;
        static const size_t feature_local         = 1 << 30;
        static const size_t feature_unmask        = ~(feature_global | feature_local);

        static const size_t feature_coords        = (1 << 0)  | feature_local;
        static const size_t feature_mean          = (1 << 1)  | feature_local;
        static const size_t feature_std           = (1 << 2)  | feature_local;
        static const size_t feature_histo         = (1 << 3)  | feature_local;
        static const size_t feature_entropy       = (1 << 4)  | feature_local;
        static const size_t feature_cooccurrence  = (1 << 5)  | feature_local;
        static const size_t feature_coentropy     = (1 << 6)  | feature_local;
        static const size_t feature_cohomogenity  = (1 << 7)  | feature_local;
        static const size_t feature_cocorrelation = (1 << 8)  | feature_local;
        static const size_t feature_cocontrast    = (1 << 9)  | feature_local;
        static const size_t feature_coasm         = (1 << 10) | feature_local;
        static const size_t feature_gabor         = (1 << 11) | feature_global;

        struct autoconfig
        {
            size_t features;
            size_t kmeans_clusters;
            int wnd_size;
            bool wnd_roll;
            int histo_cols;
            int cooccurrence_cols;
            int gabor_thetas;
            int gabor_lambdas;
        };

        static autoconfig make_default_config()
        {
            return
            {
                feature_mean,
                2,
                8,
                false,
                8,
                8,
                4,
                4
            };
        }
    private:
        const parameters & p;
    public:
        segmentation_helper(const parameters & p)
            : p(p)
        {
        }
    public:
        void autoprocess(const bitmap & src, bitmap & featmap, bitmap & mask, bitmap & dst, const autoconfig & cfg) const
        {
            size_t wrc = src.mat.rows / cfg.wnd_size;
            size_t wcc = src.mat.cols / cfg.wnd_size;

            if ((cfg.features & feature_global) &&
                (cfg.features & feature_local) &&
                !cfg.wnd_roll)
            {
                return;
            }

            size_t global_feature_size = get_features_size(cfg, feature_global);

            cv::Mat samples((cfg.wnd_roll || (cfg.features & feature_global)) ?
                            ((src.mat.rows - cfg.wnd_size) * (src.mat.cols - cfg.wnd_size)) :
                            (wrc * wcc), get_features_size(cfg, feature_global | feature_local), CV_32F);
            samples.setTo(0);

            if (cfg.features & feature_global)
            {
                cv::Mat srcMat = src.mat.rowRange(cfg.wnd_size / 2, src.mat.rows - cfg.wnd_size / 2)
                                        .colRange(cfg.wnd_size / 2, src.mat.cols - cfg.wnd_size / 2);
                collect_global_features(srcMat, samples.colRange(0, global_feature_size), cfg);
            }
            
            if (cfg.features & feature_local)
            {
                if (cfg.wnd_roll)
                {
                    for (size_t r = 0; r < src.mat.rows - cfg.wnd_size; ++r)
                    for (size_t c = 0; c < src.mat.cols - cfg.wnd_size; ++c)
                    {
                        auto wnd = src.mat.rowRange(r, r + cfg.wnd_size).colRange(c, c + cfg.wnd_size);

                        collect_local_features(r, c, wnd, samples.row(r * (src.mat.cols - cfg.wnd_size) + c).colRange(global_feature_size, samples.cols), cfg);
                    }
                }
                else
                {
                    for (size_t r = 0; r < wrc; ++r)
                    for (size_t c = 0; c < wcc; ++c)
                    {
                        auto wnd = src.mat.rowRange(r * cfg.wnd_size, (r + 1) * cfg.wnd_size)
                                          .colRange(c * cfg.wnd_size, (c + 1) * cfg.wnd_size);

                        collect_local_features(r, c, wnd, samples.row(r * wcc + c).colRange(global_feature_size, samples.cols), cfg);
                    }
                }
            }

            normalize_features(samples);

            cv::Mat labels;
            double r = cv::kmeans(samples, cfg.kmeans_clusters, labels,
                                  cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 1000, 0.01),
                                  20, cv::KMEANS_RANDOM_CENTERS);

            cv::Mat flatfeats;
            pca(samples, flatfeats);

            std::vector < cv::Vec3f > cluster_colors(cfg.kmeans_clusters + 1);
            for (size_t i = 0; i < cfg.kmeans_clusters + 1; ++i)
            {
                cluster_colors[i][0] = rand() / (RAND_MAX + 1.0) * 0.8 + 0.1;
                cluster_colors[i][1] = rand() / (RAND_MAX + 1.0) * 0.8 + 0.1;
                cluster_colors[i][2] = rand() / (RAND_MAX + 1.0) * 0.8 + 0.1;
            }

            if (cfg.wnd_roll || (cfg.features & feature_global))
            {
                featmap.mat.create(src.mat.rows - cfg.wnd_size, src.mat.cols - cfg.wnd_size, CV_32F);
                mask.mat.create(src.mat.rows - cfg.wnd_size, src.mat.cols - cfg.wnd_size, CV_32FC3);
                for (size_t r = 0; r < src.mat.rows - cfg.wnd_size; ++r)
                for (size_t c = 0; c < src.mat.cols - cfg.wnd_size; ++c)
                {
                    featmap.mat.at < float > (r, c) = flatfeats.at < float > (r * (src.mat.cols - cfg.wnd_size) + c, 0);
                    int cluster = labels.at < int > (r * (src.mat.cols - cfg.wnd_size) + c, 0);
                    mask.mat.at < cv::Vec3f > (r, c) = cluster_colors[cluster];
                }
            }
            else if (cfg.features & feature_local)
            {
                featmap.mat.create(wrc, wcc, CV_32F);
                mask.mat.create(src.mat.size(), CV_32FC3);
                for (size_t r = 0; r < wrc; ++r)
                for (size_t c = 0; c < wcc; ++c)
                {
                    featmap.mat.at < float > (r, c) = flatfeats.at < float > (r * wcc + c);
                    auto wnd = mask.mat.rowRange(r * cfg.wnd_size, (r + 1) * cfg.wnd_size)
                                       .colRange(c * cfg.wnd_size, (c + 1) * cfg.wnd_size);

                    int cluster = labels.at < int > (r * wcc + c, 0);
                    wnd.setTo(cluster_colors[cluster]);
                }
                mask.mat = mask.mat.rowRange(cfg.wnd_size / 2, src.mat.rows - cfg.wnd_size / 2)
                                   .colRange(cfg.wnd_size / 2, src.mat.cols - cfg.wnd_size / 2);
            }

            double alpha = 0.6;

            cv::Mat foremask = mask.mat * alpha;
            cv::Mat background = src.mat.rowRange(cfg.wnd_size / 2, src.mat.rows - cfg.wnd_size / 2)
                                        .colRange(cfg.wnd_size / 2, src.mat.cols - cfg.wnd_size / 2);
            background = background.clone();
            background *= 1 - alpha;
            cv::cvtColor(background, background, CV_GRAY2BGR);

            dst.mat = background + foremask;
        }
    private:
        int get_features_size(const autoconfig & cfg, size_t mask) const
        {
            size_t unmasked = cfg.features & feature_unmask;

            int count = 0;

            if (feature_local & mask)
            {
                if (unmasked & feature_coords) count += 2;
                if (unmasked & feature_mean) count += 1;
                if (unmasked & feature_std) count += 1;
                if (unmasked & feature_histo) count += cfg.histo_cols;
                if (unmasked & feature_entropy) count += 1;
                if (unmasked & feature_cooccurrence) count += cfg.cooccurrence_cols * cfg.cooccurrence_cols;
                if (unmasked & feature_coentropy) count += 1;
                if (unmasked & feature_cocontrast) count += 1;
                if (unmasked & feature_cohomogenity) count += 1;
                if (unmasked & feature_cocorrelation) count += 1;
                if (unmasked & feature_coasm) count += 1;
            }

            if (feature_global & mask)
            {
                if (unmasked & feature_gabor) count += cfg.gabor_lambdas * cfg.gabor_thetas;
            }

            return count;
        }
        void collect_global_features(const cv::Mat & src, cv::Mat & dst, const autoconfig & cfg) const
        {
            double lmin = 4 / std::sqrt(2);
            int n = std::floor(std::log2(std::hypot(src.rows, src.cols) / lmin));
            for (size_t l = 0; l < cfg.gabor_lambdas; ++l)
            for (size_t t = 0; t < cfg.gabor_thetas; ++t)
            {
                int feature_id = 0;

                double lambda = 1.0 * ((1 << l) * lmin);

                double sigma = 0.5 * lambda;

                cv::Mat res;
                cv::Mat ker = cv::getGaborKernel(
                    cv::Size(),
                    1, t * CV_PI / cfg.gabor_thetas, lambda, 0.5);
                cv::filter2D(src, res, CV_32F, ker);

                res = cv::abs(res);

                cv::GaussianBlur(res, res, cv::Size(), 3 * sigma);

                for (size_t i = 0; i < src.rows; ++i)
                for (size_t j = 0; j < src.cols; ++j)
                {
                    dst.at < float > (i * src.cols + j, feature_id) = res.at < float > (i, j);
                }

                ++feature_id;
            }
        }
        void collect_local_features(size_t r, size_t c, const cv::Mat & src, cv::Mat & dstVct, const autoconfig & cfg) const
        {
            size_t unmasked = cfg.features & feature_unmask;

            int feature_id = 0;

            if (unmasked & feature_coords) 
            {
                dstVct.at < float > (feature_id++) = r;
                dstVct.at < float > (feature_id++) = c;
            }

            if (unmasked & (feature_mean | feature_std | feature_entropy | feature_histo))
            {
                double mean = 0, var = 0, entropy = 0;
                std::vector < double > histo(cfg.histo_cols + 1, 0);

                for (size_t i = 0; i < src.rows; ++i)
                for (size_t j = 0; j < src.cols; ++j)
                {
                    mean += src.at < float > (i, j);
                    var  += src.at < float > (i, j) * src.at < float > (i, j);
                    histo[(int) std::floor(src.at < float > (i, j) * cfg.histo_cols)] += 1;
                }

                mean /= src.rows * src.cols;
                var /= src.rows * src.cols;

                var -= mean * mean;
                var = std::sqrt(std::abs(var));

                double hsum = 0;
                for (size_t i = 0; i < cfg.histo_cols; ++i) hsum += histo[i];
                for (size_t i = 0; i < cfg.histo_cols; ++i)
                {
                    histo[i] /= hsum;
                    if (!std::isfinite(histo[i])) histo[i] = 0;

                    double logval = std::log(histo[i]);
                    if (std::isfinite(logval)) entropy -= histo[i] * logval;

                    if (unmasked & feature_histo)
                    {
                        dstVct.at < float > (feature_id++) = histo[i];
                    }
                }

                entropy /= cfg.histo_cols;

                if (unmasked & feature_mean)    dstVct.at < float > (feature_id++) = mean;
                if (unmasked & feature_std)     dstVct.at < float > (feature_id++) = var;
                if (unmasked & feature_entropy) dstVct.at < float > (feature_id++) = entropy;
            }

            if (unmasked & (feature_cooccurrence | feature_coasm | feature_coentropy | feature_cohomogenity | feature_cocorrelation | feature_cocontrast))
            {
                std::vector < std::vector < double > > comatrix;
                comatrix.resize(cfg.cooccurrence_cols + 1, std::vector < double > (cfg.cooccurrence_cols + 1));

                for (size_t i = 0; i < src.rows - 1; ++i)
                for (size_t j = 0; j < src.cols - 1; ++j)
                {
                    int s1 = (int) std::floor(src.at < float > (i, j) * cfg.cooccurrence_cols);
                    int s2 = (int) std::floor(src.at < float > (i + 1, j + 1) * cfg.cooccurrence_cols);
                    comatrix[s1][s2] += 1;
                }

                double cosum = 0;
                for (size_t i = 0; i < cfg.cooccurrence_cols; ++i)
                for (size_t j = 0; j < cfg.cooccurrence_cols; ++j)
                    cosum += comatrix[i][j];
                
                double comeanx = 0, comeany = 0, costdx = 0, costdy = 0;
                double coasm = 0, cohomogenity = 0, cocontrast = 0, coentropy = 0, cocorrelation = 0;

                for (size_t i = 0; i < cfg.cooccurrence_cols; ++i)
                for (size_t j = 0; j < cfg.cooccurrence_cols; ++j)
                {
                    comatrix[i][j] /= cosum;
                    if (!std::isfinite(comatrix[i][j])) comatrix[i][j] = 0;

                    comeanx += i * comatrix[i][j];
                    comeany += j * comatrix[i][j];

                    coasm += comatrix[i][j] * comatrix[i][j];
                    cocontrast += ((int) i - (int) j) * ((int) i - (int) j) * comatrix[i][j];
                    cohomogenity += 1.0 / (1 + ((int) i - (int) j) * ((int) i - (int) j)) * comatrix[i][j];

                    double logval = std::log(comatrix[i][j]);
                    if (std::isfinite(logval)) coentropy -= comatrix[i][j] * logval;

                    if (unmasked & feature_cooccurrence)
                    {
                        dstVct.at < float > (feature_id++) = comatrix[i][j];
                    }
                }

                coasm = std::sqrt(coasm);

                if (unmasked & feature_coasm)        dstVct.at < float > (feature_id++) = coasm;
                if (unmasked & feature_cocontrast)   dstVct.at < float > (feature_id++) = cocontrast;
                if (unmasked & feature_cohomogenity) dstVct.at < float > (feature_id++) = cohomogenity;
                if (unmasked & feature_coentropy)    dstVct.at < float > (feature_id++) = coentropy;

                if (unmasked & feature_cocorrelation)
                {
                    for (size_t i = 0; i < cfg.cooccurrence_cols; ++i)
                    for (size_t j = 0; j < cfg.cooccurrence_cols; ++j)
                    {
                        costdx += ((int) i - comeanx) * ((int) i - comeanx) * comatrix[i][j];
                        costdy += ((int) j - comeany) * ((int) j - comeany) * comatrix[i][j];
                    }

                    costdx = std::sqrt(costdx);
                    costdy = std::sqrt(costdy);

                    for (size_t i = 0; i < cfg.cooccurrence_cols; ++i)
                    for (size_t j = 0; j < cfg.cooccurrence_cols; ++j)
                    {
                        cocorrelation += ((int) i - comeanx) * ((int) j - comeany) / costdx / costdy * comatrix[i][j];
                    }

                    if (!std::isfinite(cocorrelation)) cocorrelation = 0;

                    dstVct.at < float > (feature_id++) = cocorrelation;
                }
            }
        }
        void normalize_features(cv::Mat & samples) const
        {
            cv::Mat moments(2, samples.cols, CV_32F);
            moments.setTo(0);
            
            for (size_t i = 0; i < samples.rows; ++i)
            {
                moments.row(0) += samples.row(i);
                moments.row(1) += samples.row(i).mul(samples.row(i));
            }
            
            moments /= samples.rows;
            
            moments.row(1) -= moments.row(0).mul(moments.row(0));
            
            moments.row(1) = cv::abs(moments.row(1));
            cv::sqrt(moments.row(1), moments.row(1));
            
            for (size_t i = 0; i < samples.rows; ++i)
            {
                samples.row(i) -= moments.row(0);
                samples.row(i) /= moments.row(1);
            }
        }
        void pca(const cv::Mat & samples, cv::Mat & featmap) const
        {
            cv::PCA pca(samples, cv::noArray(), cv::PCA::DATA_AS_ROW, 1);
            featmap = pca.project(samples);
            double imin, imax;
            cv::minMaxIdx(featmap, &imin, &imax);
            featmap = (featmap - imin) / (imax - imin);
        }
    };

    /*****************************************************/
    /*                     model                         */
    /*****************************************************/

    struct model_data
    {
        parameters params;
        CBitmap csource;
        bitmap source;
        CBitmap cnoised;
        bitmap noised;
        CBitmap cmask;
        bitmap mask;
        CBitmap cresult;
        bitmap result;
        CBitmap cfeatmap;
        bitmap featmap;
    };
}