/**
 * Unscented Kalman Filter Project
 * 
 * This file contains the main driving code for the Unscented Kalman Filter
 * 
 * Author: Morsinaldo Medeiros
 * Date:   2024/01/17
 * 
 * Implementation reference: https://github.com/mohanadhammad/sfnd-unscented-kalman-filter/blob/master/src/ukf.cpp
 * Github Copilot was used to help with the implementation
*/
#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // std_a_ = 30;
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // std_yawdd_ = 30;
  std_yawdd_ = 5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * DONE: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false;

  // initialize timestamp
  time_us_ = 0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // predicted sigma points matrix
  Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = Eigen::VectorXd(2 * n_aug_ + 1);

  // set weights
  const double d{lambda_ + n_aug_};
  weights_(0) = lambda_ / d;

  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5 / d;
  }

}

// Destructor
UKF::~UKF() {}

// Normalize angle
void UKF::NormalizeAngle(double &angle) const {
  angle = fmod(angle + M_PI, 2 * M_PI);

  // force angle in range [0, 2 * M_PI)
  if (angle < 0) {
    angle += 2 * M_PI;
  }

  angle -= M_PI;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * DONE: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_){
    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER){
      // Initialize state.
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
      is_initialized_ = true;
    }
    else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR){
      // Convert radar from polar to cartesian coordinates and initialize state.
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double drho = meas_package.raw_measurements_(2);

      // Convert from polar to cartesian
      double vx = drho * std::cos(phi);
      double vy = drho * std::sin(phi);

      // Initialize state.
      x_(0) = rho * std::cos(phi);
      x_(1) = rho * std::sin(phi);
      x_(2) = vx * std::cos(phi) + vy * std::sin(phi); // for signed velocity value

      is_initialized_ = true;
    }
    else {
      return;
    }

    // Save the initiall timestamp for dt calculation
    time_us_ = meas_package.timestamp_;
    return;
  }

  // Calculate dt
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; // dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  // Prediction step
  this->Prediction(dt);

  // Update step
  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER){
    this->UpdateLidar(meas_package);
  }
  else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR){
    this->UpdateRadar(meas_package);
  }
  else {
    return;
  }
} 

void UKF::Prediction(double delta_t) {
  /**
   * DONE: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // Create augmented mean vector
  Eigen::VectorXd x_aug = Eigen::VectorXd(n_aug_);

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  // Create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  // Create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // Create square root matrix
  Eigen::MatrixXd L_aug = P_aug.llt().matrixL();

  // Create augmented sigma points
  const double scale{std::sqrt(lambda_ + n_aug_)};

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + scale * L_aug.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - scale * L_aug.col(i);
  }

  const double delta_t2{delta_t * delta_t};

  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Extract values for better readability
    double p_x{Xsig_aug(0, i)};
    double p_y{Xsig_aug(1, i)};
    double v{Xsig_aug(2, i)};
    double psi{Xsig_aug(3, i)};
    double psi_dot{Xsig_aug(4, i)};
    double nu_a{Xsig_aug(5, i)};
    double nu_psi_dot_dot{Xsig_aug(6, i)};

    const double cos_psi{std::cos(psi)};
    const double sin_psi{std::sin(psi)};

    // Avoid division by zero
    if (std::abs(psi_dot) < std::numeric_limits<double>::epsilon()) {
      p_x += (v * cos_psi * delta_t) + (0.5 * delta_t2 * cos_psi * nu_a);
      p_y += (v * sin_psi * delta_t) + (0.5 * delta_t2 * sin_psi * nu_a);
    } else {
      p_x += (v / psi_dot) * (std::sin(psi + psi_dot * delta_t) - sin_psi) + (0.5 * delta_t2 * cos_psi * nu_a);
      p_y += (v / psi_dot) * (-std::cos(psi + psi_dot * delta_t) + cos_psi) + (0.5 * delta_t2 * sin_psi * nu_a);
    }

    v += delta_t * nu_a;
    psi += psi_dot * delta_t + 0.5 * delta_t2 * nu_psi_dot_dot;
    psi_dot += delta_t * nu_psi_dot_dot;

    // Write predicted sigma points into right column
    Xsig_pred_(0, i) = p_x;
    Xsig_pred_(1, i) = p_y;
    Xsig_pred_(2, i) = v;
    Xsig_pred_(3, i) = psi;
    Xsig_pred_(4, i) = psi_dot;
  }
    
  // Create vector for predicted state
  VectorXd x_pred = VectorXd::Zero(n_x_);

  // Create covariance matrix for prediction
  MatrixXd P_pred = MatrixXd::Zero(n_x_, n_x_);

  // Predict state mean
  for (long i = 0; i < Xsig_pred_.cols(); ++i) {
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }

  // Normalize angle
  NormalizeAngle(x_(3));

  // Predict state covariance matrix
  for (long i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;

    // Normalize angle
    NormalizeAngle(x_diff(3));

    P_pred += weights_(i) * x_diff * x_diff.transpose();
  }

  // Update state and covariance matrix
  x_ = x_pred;
  P_ = P_pred;

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * DONE: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  const long n_z = meas_package.raw_measurements_.size();

  // Create matrix for sigma points in measurement space
  Eigen::VectorXd z = Eigen::VectorXd(n_z);
  z(0) = meas_package.raw_measurements_(0);
  z(1) = meas_package.raw_measurements_(1);

  // Create matrix for covariance matrix R
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  // Create matrix for measurement H
  Eigen::MatrixXd H = Eigen::MatrixXd(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  // Create matrix for predicted measurement
  Eigen::VectorXd z_pred = H * x_;
  Eigen::VectorXd z_diff = z - z_pred;

  // Create matrix for Kalman gain
  Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

  // New estimate
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_x_, n_x_);
  x_ += K * z_diff;
  P_ = (I - K * H) * P_;

  // Normalize angle
  NormalizeAngle(x_(3));
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * DONE: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  const long n_z = meas_package.raw_measurements_.size();

  // Create matrix for sigma points in measurement space
  MatrixXd z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  // Transform sigma points into measurement space
  for (long i = 0; i < Xsig_pred_.cols(); ++i) {
    // Extract values for better readability
    const double p_x{Xsig_pred_(0, i)};
    const double p_y{Xsig_pred_(1, i)};
    const double v{Xsig_pred_(2, i)};
    const double psi{Xsig_pred_(3, i)};
    
    // Measurement model
    z_sig(0, i) = std::sqrt(p_x * p_x + p_y * p_y); // rho
    z_sig(1, i) = std::atan2(p_y, p_x); // phi
    z_sig(2, i) = (p_x * std::cos(psi) * v + p_y * std::sin(psi) * v) / z_sig(0, i); // rho_dot
    
  }

  // Calculate mean predicted measurement
  for (long i = 0; i < z_sig.cols(); ++i) {
    z_pred += weights_(i) * z_sig.col(i);
  }

  // Calculate measurement covariance matrix S
  for (long i = 0; i < z_sig.cols(); ++i) {
    VectorXd z_diff = z_sig.col(i) - z_pred;

    // Normalize angle
    NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Create matrix for covariance matrix R
  MatrixXd R(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;

  // Add measurement noise covariance matrix
  S += R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // Calculate cross correlation matrix
  for (long i = 0; i < weights_.rows(); i++) {
  
    VectorXd z_diff = z_sig.col(i) - z_pred;

    // Normalize angle
    NormalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Normalize angle
    NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
  Eigen::VectorXd z = Eigen::VectorXd(n_z);
  z(0) = meas_package.raw_measurements_(0);
  z(1) = meas_package.raw_measurements_(1);
  z(2) = meas_package.raw_measurements_(2);

  VectorXd z_diff = z - z_pred;
  NormalizeAngle(z_diff(1));

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  // Normalize angle
  NormalizeAngle(x_(3));
}