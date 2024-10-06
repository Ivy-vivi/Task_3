#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "windmill.hpp"

using namespace std;
using namespace cv;

#define wm_AO 1.305
#define wm_A 0.785
#define wm_w 1.884
#define wm_phi 1.65

struct TimeAngularVelocity{
    double t;
    double angularVelocity;
};

// 识别 "R" 标记及最简单的锤子并绘制中心点
void drawCenters(cv::Mat &img, double distanceFactor, cv::Point &hammerCenter, cv::Point &rCenter)
{
    cv::Mat gray, binary;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    bool foundHammerCenter = false;
    bool foundRCenter = false;

    cv::Point tempHammerCenter, tempRCenter;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        cv::Moments m = cv::moments(contour);
        if (m.m00 != 0)
        {
            cv::Point center(m.m10 / m.m00, m.m01 / m.m00);

            if (area > 500 && area < 10000)
            {
                tempHammerCenter = center;
                foundHammerCenter = true;
            }
            if (area < 500)
            {
                tempRCenter = center;
                foundRCenter = true;
            }
        }
    }

    // 仅在找到有效的锤子中心和 "R" 标记中心时进行绘制
    if (foundHammerCenter && foundRCenter)
    {
        cv::Point vector = tempHammerCenter - tempRCenter;
        double length = std::sqrt(vector.x * vector.x + vector.y * vector.y);
        double scale = (length + distanceFactor) / length;
        cv::Point newHammerCenter = tempRCenter + cv::Point(vector.x * scale, vector.y * scale);

        // 绘制新的锤子中心点
        cv::circle(img, newHammerCenter, 5, cv::Scalar(0, 255, 0), -1);

        // 绘制 "R" 标记的中心点
        cv::circle(img, tempRCenter, 5, cv::Scalar(0, 255, 0), -1);

        hammerCenter = newHammerCenter;
        rCenter = tempRCenter;
    }
}
struct WindmillResidual
{
    WindmillResidual(double t, double v) : t_(t), v_(v) {}

    template <typename T>
    bool operator()(const T *const params, T *residual) const
    {
        const T& A0 = params[0];
        const T& A = params[1];
        const T& omega = params[2];
        const T& phi = params[3];

        T predicted = A0 + A * ceres::cos(omega * t_ + phi);
        T error = v_ - predicted;

        residual[0] = error;

        return true;
    }

private:
    const double t_;
    const double v_;
};


bool isConverged(const double* params)
{
    if(abs(wm_AO - params[0]) > 0.05 * abs(params[0])){
        return false;
    }
    if(abs(wm_A - params[1]) > 0.05 * abs(params[1])){
        return false;
    }
    if(abs(wm_w - params[2]) > 0.05 * abs(params[2])){
        return false;
    }
    if(abs(wm_phi - params[3]) > 0.05 * abs(params[3])){
        return false;
    }
    return false;
}

//To calculate the angle between two vectors RA at t and t+1
double calculate_angle(const cv::Point2f& A_t, const cv::Point2f& A_t1, const cv::Point2f& R_t, const cv::Point2f& R_t1) {
    // Vectors RA at time t and t+1
    cv::Point2f RA_t = A_t - R_t;
    cv::Point2f RA_t1 = A_t1 - R_t1;
    
    // Dot product of vectors RA_t and RA_t1
    double dot_product = RA_t.x * RA_t1.x + RA_t.y * RA_t1.y;
    
    // Magnitudes of the vectors
    double magnitude_RA_t = std::sqrt(RA_t.x * RA_t.x + RA_t.y * RA_t.y);
    double magnitude_RA_t1 = std::sqrt(RA_t1.x * RA_t1.x + RA_t1.y * RA_t1.y);
    
    // Calculate the cosine of the angle
    double cos_theta = dot_product / (magnitude_RA_t * magnitude_RA_t1);
    
    // Ensure the cosine value is within the valid range [-1, 1] to avoid floating point errors
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    
    //the angle in radians
    double radians = std::acos(cos_theta);
    // double degress = radians / 3.1415926 * 180;
    // degress = fmod(degress, 360.0);

    // if (degress < 0) {
    //     degress += 360.0;
    // }
    // return degress;
    return radians;
}

double calculateAngularVelocity(double t0, double dt)
{
    double dangle = wm_AO * dt + (wm_A / wm_w) * (cos(wm_w * t0 + wm_phi) - cos(wm_w * (t0 + dt) + wm_phi));
    //double angular = dangle / 3.1415926 * 180;
    //double angular_velocity = angular / dt; 

    return dangle / dt;
}

double calculateRadians(double t0, double dt)
{
    double dangle = wm_AO * dt + (wm_A / wm_w) * (cos(wm_w * t0 + wm_phi) - cos(wm_w * (t0 + dt) + wm_phi));
    return dangle;
}


int main()
{
    double t_sum = 0;
    const int N = 10;
    for (int num = 0; num < N; num++)
    {
        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t_start = (double)t.count() / 1000;
        WINDMILL::WindMill wm(0);
        Mat src;

        //ceres::Problem problem;
        double A0 = 0.305, A = 1.785, w = 0.884, phi = 1.24;

        int count = 0;

        // starttime
        int64 start_time = getTickCount();

        //render parameter
        double distanceFactor = 0;

        //solver parameter
        std::vector<TimeAngularVelocity> dataset;

        double prev_time = t_start;
        cv::Point prev_hammerCenter;
        cv::Point prev_rCenter;
        double params[4] = {A0, A, w, phi};
        bool first_frame = true;

        while (1)
        {
            t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            double t_now = (double)t.count() / 1000;
            src = wm.getMat(t_now - t_start); 
            // 控制距离因子
            double distanceFactor = 17.0;  
            //render center
            cv::Point hammerCenter, rCenter;
            drawCenters(src, distanceFactor, hammerCenter, rCenter);

            if(!first_frame){
                //construct dataset
                double theta = calculate_angle(prev_hammerCenter, hammerCenter, prev_rCenter, rCenter);
                double dt = (t_now - prev_time);
                double angularVelocity = theta / dt;

                //double radians = calculateRadians(t_now, dt);
                //printf("(%f, %f, %f)\n", theta, radians, abs(radians - theta));

                // double v = calculateAngularVelocity((t_now - t_start), (t_now - t_start));
                // printf("(%f, %f, %f)\n", angularVelocity, v, abs(v - angularVelocity));

                dataset.push_back({(t_now - t_start), angularVelocity});
            }else{
                dataset.push_back({0, 0});
                first_frame = false;
            }

            prev_time = t_now;
            prev_hammerCenter = hammerCenter;
            prev_rCenter = rCenter;

            //solve
            if(dataset.size() > 300){
                ceres::Problem problem;
                for (size_t i = 0; i < dataset.size(); ++i)
                {
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<WindmillResidual, 1, 4>(
                            new WindmillResidual(dataset[i].t, dataset[i].angularVelocity)),
                        nullptr, (double*)params);
                }
                ceres::Solver::Options options;
                options.function_tolerance = 1e-6; //change to 0.05
                //options.function_tolerance = 0.05;
                options.parameter_tolerance = 1e-8;
                options.max_num_iterations = 100;
                options.linear_solver_type = ceres::DENSE_QR;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                if (summary.termination_type == ceres::CONVERGENCE) {
                    //std::cout << summary.FullReport() << std::endl;
                    std::cout << summary.BriefReport() << std::endl;
                    t_sum += (getTickCount() - start_time);
                    //printf("%ld,(%2f,%2f,%2f,%2f)\n",(getTickCount() - start_time), params[0],params[1],params[2],params[3]);
                    std::cout << "The solution has converged!" << std::endl;
                    break;
                } else if (summary.termination_type == ceres::NO_CONVERGENCE) {
                    std::cout << "The solution did not converge." << std::endl;
                }
                //check parameter 5%
                // if(isConverged((double*)params)){
                //     std::cout << summary.FullReport() << std::endl;
                //     t_sum += (getTickCount() - start_time);
                //     break;
                // }
                // printf("%f,(%2f,%2f,%2f,%2f)\n",t_now, params[0],params[1],params[2],params[3]);
            }

            imshow("windmill", src);
            waitKey(1);
        }
    }
    std::cout << "cost(s): " << (t_sum / getTickFrequency()) / N << std::endl;
}