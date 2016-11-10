subroutine shellBucklingEurocode(nPoints, d, t, sigma_z, sigma_t, tau_zt, L_reinforced, E, sigma_y, &
  &gamma_f, gamma_b, EU_utilization)
  ! Estimate shell buckling utilization along tower.

  ! Arguments:
  ! npt - number of locations at each node at which stress is evaluated.
  ! sigma_z - axial stress at npt*node locations.  must be in order
  !               [(node1_pts1-npt), (node2_pts1-npt), ...]
  ! sigma_t - azimuthal stress given at npt*node locations
  ! tau_zt - shear stress (z, theta) at npt*node locations
  ! E - modulus of elasticity
  ! sigma_y - yield stress
  ! L_reinforced - reinforcement length - structure is re-discretized with this spacing
  ! gamma_f - safety factor for stresses
  ! gamma_b - safety factor for buckling
  !
  ! Returns:
  ! z
  ! EU_utilization: - array of shell buckling utilizations evaluted at (z[0] at npt locations, \n
  !                   z[0]+L_reinforced at npt locations, ...). \n
  !                   Each utilization must be < 1 to avoid failure.

  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  ! in
  integer, intent(in) :: nPoints
  real(dp), intent(in) :: gamma_f, gamma_b
  real(dp), dimension(nPoints), intent(in) :: d, t, sigma_z, sigma_t, L_reinforced, E, sigma_y
  real(dp), dimension(nPoints), intent(in) :: tau_zt

  !out
  real(dp), dimension(nPoints), intent(out) :: EU_utilization

  !local
  integer :: i
  real(dp) :: h, r1, r2, t1, t2, sigma_z_shell, sigma_t_shell, tau_zt_shell, utilization
  real(dp), dimension(nPoints) :: sigma_z_sh, sigma_t_sh, tau_zt_sh

  do i = 1, nPoints
    h = L_reinforced(i)

    r1 = d(i)/2.0_dp - t(i)/2.0_dp
    r2 = d(i)/2.0_dp - t(i)/2.0_dp
    t1 = t(i)
    t2 = t(i)

    sigma_z_shell = sigma_z(i)
    sigma_t_shell = sigma_t(i)
    tau_zt_shell = tau_zt(i)

    ! TO DO: the following is non-smooth, although in general its probably OK
    ! change to magnitudes and add safety factor
    sigma_z_shell = gamma_f*abs(sigma_z_shell)
    sigma_t_shell = gamma_f*abs(sigma_t_shell)
    tau_zt_shell = gamma_f*abs(tau_zt_shell)

    call shellBucklingOneSection(h, r1, r2, t1, t2, gamma_b, sigma_z_shell, sigma_t_shell, &
                                &tau_zt_shell, E(i), sigma_y(i), utilization)
    EU_utilization(i) = utilization

    ! make them into vectors
    ! TODO is this necessary?
    sigma_z_sh(i)=sigma_z_shell
    sigma_t_sh(i)=sigma_t_shell
    tau_zt_sh(i)=tau_zt_shell

  end do

end subroutine shellBucklingEurocode


subroutine shellBucklingOneSection(h, r1, r2, t1, t2, gamma_b, sigma_z, sigma_t, tau_zt, E, sigma_y, utilization)

  ! Estimate shell buckling for one tapered cylindrical shell section.
  !
  ! Arguments:
  ! h - height of conical section
  ! r1 - radius at bottom
  ! r2 - radius at top
  ! t1 - shell thickness at bottom
  ! t2 - shell thickness at top
  ! E - modulus of elasticity
  ! sigma_y - yield stress
  ! gamma_b - buckling reduction safety factor
  ! sigma_z - axial stress component
  ! sigma_t - azimuthal stress component
  ! tau_zt - shear stress component (z, theta)
  !
  ! Returns:
  ! EU_utilization, shell buckling utilization which must be < 1 to avoid failure

  ! NOTE: definition of r1, r2 switched from Eurocode document to be consistent with FEM.


  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  !in
  real(dp), intent(in) :: h, r1, r2, t1, t2, gamma_b, sigma_z, sigma_t, tau_zt, E, sigma_y

  !out
  real(dp), intent(out) :: utilization

  !local
  real(dp) :: beta, L, t, le, re, omega, rovert, Cx, sigma_z_Rcr, lambda_z0, beta_z, &
              &eta_z, Q, lambda_z, delta_wk, alpha_z, chi_z, sigma_z_Rk, sigma_z_Rd, &
              &sigma_t_Rcr, alpha_t, lambda_t0, beta_t, eta_t, lambda_t, chi_theta, &
              &sigma_t_Rk, sigma_t_Rd, rho, C_tau, tau_zt_Rcr, alpha_tau, beta_tau, &
              &lambda_tau0, eta_tau, lambda_tau, chi_tau, tau_zt_Rk, tau_zt_Rd, k_z, &
              &k_theta, k_tau, k_i

  ! ----- geometric parameters --------
  beta = atan2(r1-r2, h)
  L = h/cos(beta)
  t = 0.5_dp*(t1+t2)

  ! ------------- axial stress -------------
  ! length parameter
  le = L
  re = 0.5_dp*(r1+r2)/cos(beta)
  omega = le/sqrt(re*t)
  rovert = re/t

  ! compute Cx
  call cxsmooth(omega, rovert, Cx)


  ! if omega <= 1.7:
  !     Cx = 1.36 - 1.83/omega + 2.07/omega/omega
  ! elif omega > 0.5*rovert:
  !     Cxb = 6.0  ! clamped-clamped
  !     Cx = max(0.6, 1 + 0.2/Cxb*(1-2.0*omega/rovert))
  ! else:
  !     Cx = 1.0

  ! critical axial buckling stress
  sigma_z_Rcr = 0.605_dp*E*Cx/rovert

  ! compute buckling reduction factors
  lambda_z0 = 0.2_dp
  beta_z = 0.6_dp
  eta_z = 1.0_dp
  Q = 25.0_dp  ! quality parameter - high
  lambda_z = sqrt(sigma_y/sigma_z_Rcr)
  delta_wk = 1.0_dp/Q*sqrt(rovert)*t
  alpha_z = 0.62_dp/(1.0_dp + 1.91_dp*(delta_wk/t)**1.44)

  call buckling_reduction_factor(alpha_z, beta_z, eta_z, lambda_z0, lambda_z, chi_z)

  ! design buckling stress
  sigma_z_Rk = chi_z*sigma_y
  sigma_z_Rd = sigma_z_Rk/gamma_b

  ! ---------------- hoop stress ------------------

  ! length parameter
  le = L
  re = 0.5_dp*(r1+r2)/(cos(beta))
  omega = le/sqrt(re*t)
  rovert = re/t

  ! Ctheta = 1.5  ! clamped-clamped
  ! CthetaS = 1.5 + 10.0/omega**2 - 5.0/omega**3

  ! ! critical hoop buckling stress
  ! if (omega/Ctheta < 20.0):
  !     sigma_t_Rcr = 0.92*E*CthetaS/omega/rovert
  ! elif (omega/Ctheta > 1.63*rovert):
  !     sigma_t_Rcr = E*(1.0/rovert)**2*(0.275 + 2.03*(Ctheta/omega*rovert)**4)
  ! else:
  !     sigma_t_Rcr = 0.92*E*Ctheta/omega/rovert

  call sigmasmooth(omega, E, rovert, sigma_t_Rcr)

  ! buckling reduction factor
  alpha_t = 0.65_dp  ! high fabrication quality
  lambda_t0 = 0.4_dp
  beta_t = 0.6_dp
  eta_t = 1.0_dp
  lambda_t = sqrt(sigma_y/sigma_t_Rcr)

  call buckling_reduction_factor(alpha_t, beta_t, eta_t, lambda_t0, lambda_t, chi_theta)

  sigma_t_Rk = chi_theta*sigma_y
  sigma_t_Rd = sigma_t_Rk/gamma_b

  ! ----------------- shear stress ----------------------

  ! length parameter
  le = h
  rho = sqrt((r1+r2)/(2.0_dp*r2))
  re = (1.0_dp + rho - 1.0_dp/rho)*r2*cos(beta)
  omega = le/sqrt(re*t)
  rovert = re/t

  ! if (omega < 10):
  !     C_tau = sqrt(1.0 + 42.0/omega**3)
  ! elif (omega > 8.7*rovert):
  !     C_tau = 1.0/3.0*sqrt(omega/rovert)
  ! else:
  !     C_tau = 1.0
  call tausmooth(omega, rovert, C_tau)

  tau_zt_Rcr = 0.75_dp*E*C_tau*sqrt(1.0_dp/omega)/rovert

  ! reduction factor
  alpha_tau = 0.65_dp  ! high fabrifaction quality
  beta_tau = 0.6_dp
  lambda_tau0 = 0.4_dp
  eta_tau = 1.0_dp
  lambda_tau = sqrt(sigma_y/sqrt(3.0_dp)/tau_zt_Rcr)

  call buckling_reduction_factor(alpha_tau, beta_tau, eta_tau, lambda_tau0, lambda_tau, chi_tau)

  tau_zt_Rk = chi_tau*sigma_y/sqrt(3.0_dp)
  tau_zt_Rd = tau_zt_Rk/gamma_b

  ! buckling interaction parameters

  k_z = 1.25_dp + 0.75_dp*chi_z
  k_theta = 1.25_dp + 0.75_dp*chi_theta
  k_tau = 1.75_dp + 0.25_dp*chi_tau
  k_i = (chi_z*chi_theta)**2

  ! shell buckling utilization

  utilization = (sigma_z/sigma_z_Rd)**k_z + (sigma_t/sigma_t_Rd)**k_theta - &
  &k_i*(sigma_z*sigma_t/sigma_z_Rd/sigma_t_Rd) + (tau_zt/tau_zt_Rd)**k_tau

end subroutine shellBucklingOneSection


subroutine cxsmooth(omega, rovert, Cx)

  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  !in
  real(dp), intent(in) :: omega, rovert

  !out
  real(dp), intent(out) :: Cx

  !local
  real(dp) :: Cxb, constant, ptL1, ptR1, ptL2, ptR2, ptL3, ptR3, fL, fR, gL, gR

  !evaluate the function
  Cxb = 6.0_dp  ! clamped-clamped
  constant = 1.0_dp + 1.83_dp/1.7_dp - 2.07_dp/1.7_dp**2

  ptL1 = 1.7_dp-0.25_dp
  ptR1 = 1.7_dp+0.25_dp

  ptL2 = 0.5_dp*rovert - 1.0_dp
  ptR2 = 0.5_dp*rovert + 1.0_dp

  ptL3 = (0.5_dp+Cxb)*rovert - 1.0_dp
  ptR3 = (0.5_dp+Cxb)*rovert + 1.0_dp


  if (omega < ptL1) then
    Cx = constant - 1.83_dp/omega + 2.07_dp/omega**2

  else if (omega >= ptL1 .and. omega <= ptR1) then

    fL = constant - 1.83_dp/ptL1 + 2.07_dp/ptL1**2
    fR = 1.0_dp
    gL = 1.83_dp/ptL1**2 - 4.14_dp/ptL1**3
    gR = 0.0_dp

    call cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega, Cx)

  else if (omega > ptR1 .and. omega < ptL2) then
    Cx = 1.0_dp

  else if (omega >= ptL2 .and. omega <= ptR2) then

    fL = 1.0_dp
    fR = 1.0_dp + 0.2_dp/Cxb*(1.0_dp-2.0_dp*ptR2/rovert)
    gL = 0.0_dp
    gR = -0.4_dp/Cxb/rovert

    call cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega, Cx)

  else if (omega > ptR2 .and. omega < ptL3) then
    Cx = 1.0_dp + 0.2_dp/Cxb*(1.0_dp-2.0_dp*omega/rovert)

  else if (omega >= ptL3 .and. omega <= ptR3) then

    fL = 1.0_dp + 0.2_dp/Cxb*(1.0_dp-2.0_dp*ptL3/rovert)
    fR = 0.6_dp
    gL = -0.4_dp/Cxb/rovert
    gR = 0.0_dp

    call cubic_spline_eval(ptL3, ptR3, fL, fR, gL, gR, omega, Cx)

  else
    Cx = 0.6_dp

  end if

end subroutine cxsmooth



subroutine sigmasmooth(omega, E, rovert, sigma)

  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  !in
  real(dp), intent(in) :: omega, E, rovert

  !out
  real(dp), intent(out) :: sigma

  !local
  real(dp) :: Ctheta, ptL, ptR, offset, Cthetas, fL, fR, gL, gR, alpha1


  Ctheta = 1.5_dp  ! clamped-clamped

  ptL = 1.63_dp*rovert*Ctheta - 1.0_dp
  ptR = 1.63_dp*rovert*Ctheta + 1.0_dp

  if (omega < 20.0_dp*Ctheta) then
    offset = (10.0_dp/(20.0_dp*Ctheta)**2 - 5.0_dp/(20.0_dp*Ctheta)**3)
    Cthetas = 1.5_dp + 10.0_dp/omega**2 - 5.0_dp/omega**3 - offset
    sigma = 0.92_dp*E*Cthetas/omega/rovert

  else if (omega >= 20.0_dp*Ctheta .and. omega < ptL) then

    sigma = 0.92_dp*E*Ctheta/omega/rovert

  else if (omega >= ptL .and. omega <= ptR) then

    alpha1 = 0.92_dp/1.63_dp - 2.03_dp/1.63_dp**4

    fL = 0.92_dp*E*Ctheta/ptL/rovert
    fR = E*(1.0_dp/rovert)**2*(alpha1 + 2.03_dp*(Ctheta/ptR*rovert)**4)
    gL = -0.92_dp*E*Ctheta/rovert/ptL**2
    gR = -E*(1.0_dp/rovert)*2.03_dp*4.0_dp*(Ctheta/ptR*rovert)**3*Ctheta/ptR**2

    call cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, omega, sigma)

  else

    alpha1 = 0.92_dp/1.63_dp - 2.03_dp/1.63_dp**4
    sigma = E*(1.0_dp/rovert)**2*(alpha1 + 2.03_dp*(Ctheta/omega*rovert)**4)

  end if

end subroutine sigmasmooth


subroutine tausmooth(omega, rovert, C_tau)

  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  !in
  real(dp), intent(in) :: omega, rovert

  !out
  real(dp), intent(out) :: C_tau

  !local
  real(dp) :: ptL1, ptR1, ptL2, ptR2, fL, fR, gL, gR


  ptL1 = 9.0_dp
  ptR1 = 11.0_dp

  ptL2 = 8.7_dp*rovert - 1.0_dp
  ptR2 = 8.7_dp*rovert + 1.0_dp

  if (omega < ptL1) then
    C_tau = sqrt(1.0_dp + 42.0_dp/omega**3 - 42.0_dp/10.0_dp**3)

  else if (omega >= ptL1 .and. omega <= ptR1) then
    fL = sqrt(1.0_dp + 42.0_dp/ptL1**3 - 42.0_dp/10.0_dp**3)
    fR = 1.0_dp
    gL = -63.0_dp/ptL1**4/fL
    gR = 0.0_dp

    call cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega, C_tau)

  else if (omega > ptR1 .and. omega < ptL2) then
    C_tau = 1.0_dp

  else if (omega >= ptL2 .and. omega <= ptR2) then
    fL = 1.0_dp
    fR = 1.0_dp/3.0_dp*sqrt(ptR2/rovert) + 1.0_dp - sqrt(8.7_dp)/3.0_dp
    gL = 0.0_dp
    gR = 1.0_dp/6.0_dp/sqrt(ptR2*rovert)

    call cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega, C_tau)

  else
    C_tau = 1.0_dp/3.0_dp*sqrt(omega/rovert) + 1.0_dp - sqrt(8.7_dp)/3.0_dp

  end if

end subroutine tausmooth


subroutine buckling_reduction_factor(alpha, beta, eta, lambda_0, lambda_bar, chi)
  ! Computes a buckling reduction factor used in Eurocode shell buckling formula.

  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  !in
  real(dp), intent(in) :: alpha, beta, eta, lambda_0, lambda_bar

  !out
  real(dp), intent(out) :: chi

  !local
  real(dp) :: lambda_p, ptL, ptR, fracR, fL, fR, gL, gR

  lambda_p = sqrt(alpha/(1.0_dp-beta))

  ptL = 0.9_dp*lambda_0
  ptR = 1.1_dp*lambda_0

  if (lambda_bar < ptL) then
    chi = 1.0_dp

  else if (lambda_bar >= ptL .and. lambda_bar <= ptR) then  ! cubic spline section

    fracR = (ptR-lambda_0)/(lambda_p-lambda_0)
    fL = 1.0_dp
    fR = 1.0_dp-beta*fracR**eta
    gL = 0.0_dp
    gR = -beta*eta*fracR**(eta-1)/(lambda_p-lambda_0)

    call cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, lambda_bar, chi)

  else if (lambda_bar > ptR .and. lambda_bar < lambda_p) then
    chi = 1.0_dp - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

  else
    chi = alpha/lambda_bar**2

  end if

  ! if (lambda_bar <= lambda_0):
  !     chi = 1.0
  ! elif (lambda_bar >= lambda_p):
  !     chi = alpha/lambda_bar**2
  ! else:
  !     chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

end subroutine buckling_reduction_factor




subroutine cubic_spline_eval(x1, x2, f1, f2, g1, g2, x, poly)

  implicit none

  ! define precision to be the standard for a double precision ! on local system
  integer, parameter :: dp = kind(0.d0)

  !in
  real(dp), intent(in) :: x1, x2, f1, f2, g1, g2, x

  !out
  real(dp), intent(out) :: poly

  !local
  real(dp) :: det, B11, B12, B13, B14, B21, B22, B23, B24,&
              &B31, B32, B33, B34, B41, B42, B43, B44
  real(dp), dimension(4) :: A1, A2, A3, A4, b, coeff

  A1(1) = x1**3
  A1(2) = x1**2
  A1(3) = x1
  A1(4) = 1.0_dp

  A2(1) = x2**3
  A2(2) = x2**2
  A2(3) = x2
  A2(4) = 1.0_dp

  A3(1) = 3.0_dp*x1**2
  A3(2) = 2.0_dp*x1
  A3(3) = 1.0_dp
  A3(4) = 0.0_dp

  A4(1) = 3.0_dp*x2**2
  A4(2) = 2.0_dp*x2
  A4(3) = 1.0_dp
  A4(4) = 0.0_dp

  b(1) = f1
  b(2) = f2
  b(3) = g1
  b(4) = g2

  det = A1(1)*(A2(2)*A3(3)*A4(4)+A2(3)*A3(4)*A4(2)+A2(4)*A3(2)*A4(3)-A2(4)*A3(3)*&
        &A4(2)-A2(3)*A3(2)*A4(4)-A2(2)*A3(4)*A4(3))&
        & - A1(2)*(A2(1)*A3(3)*A4(4)+A2(3)*A3(4)*A4(1)+A2(4)*A3(1)*A4(3)-A2(4)*A3(3)*&
        &A4(1)-A2(3)*A3(1)*A4(4)-A2(1)*A3(4)*A4(3))&
        & + A1(3)*(A2(1)*A3(2)*A4(4)+A2(2)*A3(4)*A4(1)+A2(4)*A3(1)*A4(2)-A2(4)*A3(2)*&
        &A4(1)-A2(2)*A3(1)*A4(4)-A2(1)*A3(4)*A4(2))&
        & - A1(4)*(A2(1)*A3(2)*A4(3)+A2(2)*A3(3)*A4(1)+A2(3)*A3(1)*A4(2)-A2(3)*A3(2)*&
        &A4(1)-A2(2)*A3(1)*A4(3)-A2(1)*A3(3)*A4(2))

  ! entries of cof(A)
  B11 = A2(2)*A3(3)*A4(4)+A2(3)*A3(4)*A4(2)+A2(4)*A3(2)*A4(3)-A2(2)*A3(4)*A4(3)-A2(3)*A3(2)*A4(4)-A2(4)*A3(3)*A4(2)
  B12 = A1(2)*A3(4)*A4(3)+A1(3)*A3(2)*A4(4)+A1(4)*A3(3)*A4(2)-A1(2)*A3(3)*A4(4)-A1(3)*A3(4)*A4(2)-A1(4)*A3(2)*A4(3)
  B13 = A1(2)*A2(3)*A4(4)+A1(3)*A2(4)*A4(2)+A1(4)*A2(2)*A4(3)-A1(2)*A2(4)*A4(3)-A1(3)*A2(2)*A4(4)-A1(4)*A2(3)*A4(2)
  B14 = A1(2)*A2(4)*A3(3)+A1(3)*A2(2)*A3(4)+A1(4)*A2(3)*A3(2)-A1(2)*A2(3)*A3(4)-A1(3)*A2(4)*A3(2)-A1(4)*A2(2)*A3(3)

  B21 = A2(1)*A3(4)*A4(3)+A2(3)*A3(1)*A4(4)+A2(4)*A3(3)*A4(1)-A2(1)*A3(3)*A4(4)-A2(3)*A3(4)*A4(1)-A2(4)*A3(1)*A4(3)
  B22 = A1(1)*A3(3)*A4(4)+A1(3)*A3(4)*A4(1)+A1(4)*A3(1)*A4(3)-A1(1)*A3(4)*A4(3)-A1(3)*A3(1)*A4(4)-A1(4)*A3(3)*A4(1)
  B23 = A1(1)*A2(4)*A4(3)+A1(3)*A2(1)*A4(4)+A1(4)*A2(3)*A4(1)-A1(1)*A2(3)*A4(4)-A1(3)*A2(4)*A4(1)-A1(4)*A2(1)*A4(3)
  B24 = A1(1)*A2(3)*A3(4)+A1(3)*A2(4)*A3(1)+A1(4)*A2(1)*A3(3)-A1(1)*A2(4)*A3(3)-A1(3)*A2(1)*A3(4)-A1(4)*A2(3)*A3(1)

  B31 = A2(1)*A3(2)*A4(4)+A2(2)*A3(4)*A4(1)+A2(4)*A3(1)*A4(2)-A2(1)*A3(4)*A4(2)-A2(2)*A3(1)*A4(4)-A2(4)*A3(2)*A4(1)
  B32 = A1(1)*A3(4)*A4(2)+A1(2)*A3(1)*A4(4)+A1(4)*A3(2)*A4(1)-A1(1)*A3(2)*A4(4)-A1(2)*A3(4)*A4(1)-A1(4)*A3(1)*A4(2)
  B33 = A1(1)*A2(2)*A4(4)+A1(2)*A2(4)*A4(1)+A1(4)*A2(1)*A4(2)-A1(1)*A2(4)*A4(2)-A1(2)*A2(1)*A4(4)-A1(4)*A2(2)*A4(1)
  B34 = A1(1)*A2(4)*A3(2)+A1(2)*A2(1)*A3(4)+A1(4)*A2(2)*A3(1)-A1(1)*A2(2)*A3(4)-A1(2)*A2(4)*A3(1)-A1(4)*A2(1)*A3(2)

  B41 = A2(1)*A3(3)*A4(2)+A2(2)*A3(1)*A4(3)+A2(3)*A3(2)*A4(1)-A2(1)*A3(2)*A4(3)-A2(2)*A3(3)*A4(1)-A2(3)*A3(1)*A4(2)
  B42 = A1(1)*A3(2)*A4(3)+A1(2)*A3(3)*A4(1)+A1(3)*A3(1)*A4(2)-A1(1)*A3(3)*A4(2)-A1(2)*A3(1)*A4(3)-A1(3)*A3(2)*A4(1)
  B43 = A1(1)*A2(3)*A4(2)+A1(2)*A2(1)*A4(3)+A1(3)*A2(2)*A4(1)-A1(1)*A2(2)*A4(3)-A1(2)*A2(3)*A4(1)-A1(3)*A2(1)*A4(2)
  B44 = A1(1)*A2(2)*A3(3)+A1(2)*A2(3)*A3(1)+A1(3)*A2(1)*A3(2)-A1(1)*A2(3)*A3(2)-A1(2)*A2(1)*A3(3)-A1(3)*A2(2)*A3(1)

  !solve the equation Ax = b; x = A^(-1)b
  coeff(1) = (B11/det)*b(1)+(B12/det)*b(2)+(B13/det)*b(3)+(B14/det)*b(4)
  coeff(2) = (B21/det)*b(1)+(B22/det)*b(2)+(B23/det)*b(3)+(B24/det)*b(4)
  coeff(3) = (B31/det)*b(1)+(B32/det)*b(2)+(B33/det)*b(3)+(B34/det)*b(4)
  coeff(4) = (B41/det)*b(1)+(B42/det)*b(2)+(B43/det)*b(3)+(B44/det)*b(4)

  poly = coeff(1)*x**3+coeff(2)*x**2+coeff(3)*x+coeff(4)

end subroutine cubic_spline_eval
