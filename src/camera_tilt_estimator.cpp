/**
 * @file camera_tilt_estimator.cpp
 * @brief Per-camera tilt (roll/pitch) estimator from the Orbbec camera IMUs
 *
 * When the robot drives over a bump the cameras tilt, but the camera->base_link
 * transform is static (URDF), so floor points get projected as obstacles.  This
 * node estimates, for each camera, how far the camera mount is rotated away from
 * its stationary reference and publishes that rotation so consumers (the point
 * cloud combiner) can undo it per cloud.
 *
 * Estimator: complementary filter on the gravity direction, in base_link axes.
 *   - gyro propagates the estimate (a world-fixed vector seen from a rotating
 *     body moves as -w x g),
 *   - the accelerometer pulls it back with time constant comp_tau, but only when
 *     | |a| - g | < gate_tolerance (shocks are ignored),
 *   - rotation about the robot's yaw axis is removed before propagation (yaw
 *     cannot change tilt).  The yaw axis is learned per camera from the gyro
 *     while turning: the IMUs are misaligned from the URDF by up to ~2 deg, and
 *     without this a 0.4 rad/s turn leaks into a drifting roll estimate,
 *   - the robot's own acceleration from wheel odometry (dv/dt forward, v*w
 *     lateral) is subtracted first.  Without this, a 0.5 m/s^2 speed change
 *     reads as ~3 deg of pitch; a gravity-only estimator was measured injecting
 *     phantom obstacles while driving.
 *
 * Output (per camera): geometry_msgs/QuaternionStamped, frame_id = camera mount
 * frame (pivot), rotation = nominal -> actual, expressed in base_frame axes:
 *     p_corrected = o + R * (p - o),   o = mount frame origin in base_frame.
 *
 * Note: jeeves_orbbec labels IMU samples <Camera>_accel_frame, but the SDK
 * delivers them in the depth optical axes (x right, y down, z forward), so the
 * axes frame is a parameter (default <Camera>_depth_optical_frame).
 *
 * MIT License
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{
constexpr double kGravity = 9.80665;
}

class CameraTiltEstimator : public rclcpp::Node
{
public:
    CameraTiltEstimator() : Node("camera_tilt_estimator"),
                            tf_buffer_(this->get_clock()),
                            tf_listener_(tf_buffer_)
    {
        auto imu_topics = declare_parameter<std::vector<std::string>>(
            "imu_topics", {"/ob_front/camera/imu", "/ob_back/camera/imu"});
        auto mount_frames = declare_parameter<std::vector<std::string>>(
            "mount_frames", {"Front_Camera", "Back_Camera"});
        auto axes_frames = declare_parameter<std::vector<std::string>>(
            "imu_axes_frames", {"Front_Camera_depth_optical_frame", "Back_Camera_depth_optical_frame"});
        auto tilt_topics = declare_parameter<std::vector<std::string>>(
            "tilt_topics", {"/ob_front/camera/tilt", "/ob_back/camera/tilt"});
        base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
        auto odom_topic = declare_parameter<std::string>("odom_topic", "/diffbot_base_controller/odom");
        comp_tau_ = declare_parameter<double>("comp_tau", 3.0);
        gate_tolerance_ = declare_parameter<double>("gate_tolerance", 1.5);
        reference_secs_ = declare_parameter<double>("reference_secs", 2.0);
        odom_accel_tau_ = declare_parameter<double>("odom_accel_tau", 0.1);
        yaw_axis_tau_ = declare_parameter<double>("yaw_axis_tau", 5.0);
        yaw_learn_rate_ = declare_parameter<double>("yaw_learn_min_rate", 0.15);
        // Optional starting yaw axes (base axes, x y z per camera), e.g. from a previous run's
        // "yaw axis" status line; avoids roll drift during the first turns while learning.
        auto yaw_init = declare_parameter<std::vector<double>>("yaw_axes_init", std::vector<double>{});
        double publish_rate = declare_parameter<double>("publish_rate", 200.0);
        publish_period_ = publish_rate > 0.0 ? 1.0 / publish_rate : 0.0;

        if (imu_topics.size() != mount_frames.size() || imu_topics.size() != axes_frames.size() ||
            imu_topics.size() != tilt_topics.size())
        {
            throw std::invalid_argument(
                "imu_topics, mount_frames, imu_axes_frames and tilt_topics must have the same length");
        }
        if (!yaw_init.empty() && yaw_init.size() != 3 * imu_topics.size())
        {
            throw std::invalid_argument("yaw_axes_init must hold 3 values per camera");
        }

        // The driver delivers ~1 kHz IMU in bursts of ~25 samples.
        auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(200)).best_effort();
        for (size_t i = 0; i < imu_topics.size(); ++i)
        {
            auto cam = std::make_shared<Camera>();
            cam->imu_topic = imu_topics[i];
            cam->mount_frame = mount_frames[i];
            cam->axes_frame = axes_frames[i];
            if (!yaw_init.empty())
            {
                cam->yaw_axis = Eigen::Vector3d(yaw_init[3 * i], yaw_init[3 * i + 1], yaw_init[3 * i + 2]).normalized();
            }
            cam->pub = create_publisher<geometry_msgs::msg::QuaternionStamped>(tilt_topics[i], 10);
            cam->sub = create_subscription<sensor_msgs::msg::Imu>(
                imu_topics[i], imu_qos,
                [this, cam](sensor_msgs::msg::Imu::ConstSharedPtr msg) { imu_callback(*cam, *msg); });
            cameras_.push_back(cam);
        }

        if (!odom_topic.empty())
        {
            odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                odom_topic, 10,
                std::bind(&CameraTiltEstimator::odom_callback, this, std::placeholders::_1));
        }

        rezero_srv_ = create_service<std_srvs::srv::Trigger>(
            "~/rezero",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                rezero_requested_ = true;
                res->success = true;
                res->message = "Re-taking tilt reference; keep the robot still.";
            });

        stats_timer_ = create_wall_timer(std::chrono::seconds(60), [this]() { print_statistics(); });

        RCLCPP_INFO(get_logger(), "Camera tilt estimator: %zu cameras, comp_tau %.2f s, gate %.2f m/s^2, odom '%s'",
                    cameras_.size(), comp_tau_, gate_tolerance_, odom_topic.c_str());
        RCLCPP_INFO(get_logger(), "Keep the robot still for %.1f s while the reference is taken.", reference_secs_);
    }

private:
    struct Camera
    {
        std::string imu_topic, mount_frame, axes_frame;
        rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr pub;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub;
        bool have_axes{false};
        Eigen::Matrix3d R_base_imu{Eigen::Matrix3d::Identity()};
        // Reference collection
        bool have_ref{false};
        Eigen::Vector3d ref{0, 0, 1}, gyro_bias{Eigen::Vector3d::Zero()};
        Eigen::Vector3d acc_sum{Eigen::Vector3d::Zero()}, gyro_sum{Eigen::Vector3d::Zero()};
        size_t ref_count{0};
        rclcpp::Time ref_start;
        // Filter state
        Eigen::Vector3d g{0, 0, 1};
        Eigen::Vector3d yaw_axis{0, 0, 1};
        double rate{1000.0};
        size_t rate_count{0};
        rclcpp::Time rate_start;
        size_t since_publish{0};
        // Statistics
        size_t samples{0}, gated{0};
        double max_tilt_deg{0.0};
    };

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        const double t = rclcpp::Time(msg->header.stamp).seconds();
        const double v = msg->twist.twist.linear.x;
        const double w = msg->twist.twist.angular.z;
        std::lock_guard<std::mutex> lock(odom_mutex_);
        if (have_odom_)
        {
            const double dt = t - odom_t_;
            if (dt > 1e-3 && dt < 0.5)
            {
                const double a_raw = (v - odom_v_) / dt;
                const double k = dt / (odom_accel_tau_ + dt);
                accel_fwd_ += k * (a_raw - accel_fwd_);
            }
        }
        odom_t_ = t;
        odom_v_ = v;
        accel_lat_ = v * w;
        odom_w_ = w;
        have_odom_ = true;
    }

    bool lookup_axes(Camera &cam)
    {
        try
        {
            auto tf = tf_buffer_.lookupTransform(base_frame_, cam.axes_frame, tf2::TimePointZero);
            cam.R_base_imu = tf2::transformToEigen(tf.transform).rotation();
            cam.have_axes = true;
            RCLCPP_INFO(get_logger(), "%s: IMU axes taken from %s", cam.imu_topic.c_str(), cam.axes_frame.c_str());
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s: waiting for TF %s -> %s: %s",
                                 cam.imu_topic.c_str(), base_frame_.c_str(), cam.axes_frame.c_str(), ex.what());
        }
        return cam.have_axes;
    }

    void imu_callback(Camera &cam, const sensor_msgs::msg::Imu &msg)
    {
        if (!cam.have_axes && !lookup_axes(cam))
        {
            return;
        }
        if (rezero_requested_)
        {
            for (auto &c : cameras_)
            {
                c->have_ref = false;
                c->ref_count = 0;
            }
            rezero_requested_ = false;
        }

        const auto now = this->now();
        Eigen::Vector3d a = cam.R_base_imu * Eigen::Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y,
                                                              msg.linear_acceleration.z);
        const Eigen::Vector3d w = cam.R_base_imu * Eigen::Vector3d(msg.angular_velocity.x, msg.angular_velocity.y,
                                                                    msg.angular_velocity.z);

        // Samples arrive in bursts stamped with host time, so integrate the gyro
        // with the measured nominal period rather than stamp differences.
        if (cam.rate_count == 0)
        {
            cam.rate_start = now;
        }
        if (++cam.rate_count >= 2000)
        {
            const double span = (now - cam.rate_start).seconds();
            if (span > 0.0)
            {
                cam.rate = cam.rate_count / span;
            }
            cam.rate_count = 0;
        }
        const double dt = 1.0 / std::max(cam.rate, 1.0);

        if (!cam.have_ref)
        {
            if (cam.ref_count == 0)
            {
                cam.ref_start = now;
                cam.acc_sum.setZero();
                cam.gyro_sum.setZero();
            }
            cam.acc_sum += a;
            cam.gyro_sum += w;
            ++cam.ref_count;
            const double span = (now - cam.ref_start).seconds();
            if (span >= reference_secs_)
            {
                const Eigen::Vector3d mean = cam.acc_sum / cam.ref_count;
                cam.ref = mean.normalized();
                cam.gyro_bias = cam.gyro_sum / cam.ref_count;
                cam.g = cam.ref;
                cam.rate = cam.ref_count / span;
                cam.have_ref = true;
                RCLCPP_INFO(get_logger(),
                            "%s: reference taken (|a| %.3f m/s^2, absolute pitch %+.2f deg, roll %+.2f deg in %s, "
                            "%.0f Hz)",
                            cam.imu_topic.c_str(), mean.norm(), pitch_deg(cam.ref), roll_deg(cam.ref),
                            base_frame_.c_str(), cam.rate);
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            if (have_odom_)
            {
                a -= Eigen::Vector3d(accel_fwd_, accel_lat_, 0.0);
            }
        }

        // Learn the yaw axis while the robot turns (and the rotation looks like yaw,
        // so bump pitch/roll transients are not learned), then drop the yaw component.
        Eigen::Vector3d wb = w - cam.gyro_bias;
        const double wn = wb.norm();
        double odom_yaw_rate;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            odom_yaw_rate = odom_w_;
        }
        if (wn > yaw_learn_rate_ && std::abs(odom_yaw_rate) > yaw_learn_rate_)
        {
            Eigen::Vector3d u = wb / wn;
            if (u.dot(cam.yaw_axis) < 0.0)
            {
                u = -u;
            }
            if (u.dot(cam.yaw_axis) > 0.98)
            {
                const double k = dt / (yaw_axis_tau_ + dt);
                cam.yaw_axis = (cam.yaw_axis + k * (u - cam.yaw_axis)).normalized();
            }
        }
        wb -= wb.dot(cam.yaw_axis) * cam.yaw_axis;

        // Gyro propagation, then gated pull toward the measured gravity direction.
        cam.g -= wb.cross(cam.g) * dt;
        const double a_norm = a.norm();
        ++cam.samples;
        if (a_norm > 1e-3 && std::abs(a_norm - kGravity) < gate_tolerance_)
        {
            const double k = dt / (comp_tau_ + dt);
            cam.g += k * (a / a_norm - cam.g);
        }
        else
        {
            ++cam.gated;
        }
        cam.g.normalize();

        // Decimate by sample count: stamps are bursty, so a time-based gate would
        // publish the first sample of each burst instead of the latest.
        if (++cam.since_publish < std::max<size_t>(1, static_cast<size_t>(cam.rate * publish_period_)))
        {
            return;
        }
        cam.since_publish = 0;

        // Rotation taking the measured up-vector to the reference = nominal -> actual.
        const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(cam.g, cam.ref);
        const double tilt = std::acos(std::clamp(cam.g.dot(cam.ref), -1.0, 1.0)) * 180.0 / M_PI;
        cam.max_tilt_deg = std::max(cam.max_tilt_deg, tilt);

        geometry_msgs::msg::QuaternionStamped out;
        out.header.stamp = msg.header.stamp;
        out.header.frame_id = cam.mount_frame;
        out.quaternion.x = q.x();
        out.quaternion.y = q.y();
        out.quaternion.z = q.z();
        out.quaternion.w = q.w();
        cam.pub->publish(out);
    }

    static double pitch_deg(const Eigen::Vector3d &g)
    {
        return std::atan2(-g.x(), std::hypot(g.y(), g.z())) * 180.0 / M_PI;
    }

    static double roll_deg(const Eigen::Vector3d &g)
    {
        return std::atan2(g.y(), g.z()) * 180.0 / M_PI;
    }

    void print_statistics()
    {
        for (auto &cam : cameras_)
        {
            if (!cam->have_ref)
            {
                continue;
            }
            RCLCPP_INFO(get_logger(), "%s: %zu samples (%.0f Hz), %.1f%% gated, max tilt %.2f deg, now pitch %+.2f "
                        "roll %+.2f deg vs reference, yaw axis [%.3f %.3f %.3f]",
                        cam->imu_topic.c_str(), cam->samples, cam->rate,
                        cam->samples ? 100.0 * cam->gated / cam->samples : 0.0, cam->max_tilt_deg,
                        pitch_deg(cam->g) - pitch_deg(cam->ref), roll_deg(cam->g) - roll_deg(cam->ref),
                        cam->yaw_axis.x(), cam->yaw_axis.y(), cam->yaw_axis.z());
            cam->samples = cam->gated = 0;
            cam->max_tilt_deg = 0.0;
        }
    }

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string base_frame_;
    double comp_tau_, gate_tolerance_, reference_secs_, odom_accel_tau_, yaw_axis_tau_, yaw_learn_rate_,
        publish_period_;
    std::vector<std::shared_ptr<Camera>> cameras_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::mutex odom_mutex_;
    bool have_odom_{false};
    double odom_t_{0.0}, odom_v_{0.0}, odom_w_{0.0}, accel_fwd_{0.0}, accel_lat_{0.0};

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr rezero_srv_;
    std::atomic<bool> rezero_requested_{false};
    rclcpp::TimerBase::SharedPtr stats_timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraTiltEstimator>());
    rclcpp::shutdown();
    return 0;
}
