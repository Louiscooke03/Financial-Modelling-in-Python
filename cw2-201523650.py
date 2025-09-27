import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

# Question 1:
def down_and_out_milstein(T, N, r, sigma, S0, K, B, M):
    dt = T / N
    S = S0 * np.ones(M)
    V = np.zeros(M)
    barrier_crossed = np.ones(M, dtype=bool)
    
    for i in range(N):
        dw = np.sqrt(dt) * np.random.normal(0, 1, M)
        S = S + r * S * dt + sigma * S * dw + 0.5 * (sigma**2) * S * ((dw)**2 - dt)

        barrier_crossed[S <= B] = False

    V[barrier_crossed] = np.exp(-r * T) * np.maximum(S[barrier_crossed] - K, 0)
    aM = np.mean(V)
    bM = np.std(V, ddof=1)

    return aM, bM, V, S

aM, bM, V_MLSTN, S_MLSTN = down_and_out_milstein(1.5, 1000, 0.01, 0.1, 100, 90, 61, 10000)

confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)
margin_of_error = z_score * bM / np.sqrt(10000)

lower_bound = aM - margin_of_error
upper_bound = aM + margin_of_error
width_CI = upper_bound - lower_bound

# For the CI to equal $0.01 multiply by 100:
acc_M = (2* z_score * 100 * bM)**2

comp_times_MS10000 = []
for i in range(10):
    start_time_MS = time.perf_counter()
    down_and_out_milstein(1.5, 1000, 0.01, 0.1, 100, 90, 61, 10000)
    end_time_MS = time.perf_counter()
    comp_time = end_time_MS - start_time_MS
    comp_times_MS10000.append(comp_time)

compTimeMLSTN_001 = (acc_M / 10000) * np.mean(comp_times_MS10000)

print(f"The down and out option price using the Milstein Scheme is {aM}")
print(f"Confidence Interval using the Milstein Scheme is ({confidence_level * 100}%): [{lower_bound}, {upper_bound}]")
print(f"The width of the confidence interval using the Milstein Scheme is {width_CI}")
print(f"The standard deviation from using the Milstein Scheme is {bM}")
print(f"The mean compuational time for 10000 Monte-Carlo simulations using the Milstein Scheme is {np.mean(comp_times_MS10000)}")
print(f"The number of Monte-Carlo simulations required to have a CI width of $0.01 using the Milstein Scheme is {acc_M}")
print(f"The computational time required to achieve a CI = $0.01 using the Milstein Scheme is {compTimeMLSTN_001}")

# Question 2:
def down_and_out_milstein_Antithetic(T, N, r, sigma, S0, K, B, M):
    dt = T / N
    S_plus = S0 * np.ones(M)
    S_minus = S0 * np.ones(M)

    V_minus = np.zeros(M)
    V_plus = np.zeros(M)

    barrier_crossed_minus = np.ones(M, dtype=bool)
    barrier_crossed_plus = np.ones(M, dtype=bool)

    for i in range(N):
        Z = np.random.normal(0, 1, M)
        Z_antithetic = -Z
        
        dW = np.sqrt(dt) * Z
        dW_antithetic = np.sqrt(dt) * Z_antithetic

        S_plus = S_plus + (r * S_plus * dt) + (sigma * S_plus * dW) + (0.5 * (sigma**2) * S_plus * ((dW)**2 - dt))
        S_minus = S_minus + (r * S_minus * dt) + (sigma * S_minus * dW_antithetic) + (0.5 * (sigma**2) * S_minus * ((dW_antithetic)**2 - dt))

        barrier_crossed_minus[S_minus <= B] = False
        barrier_crossed_plus[S_plus <= B] = False

    V_minus[barrier_crossed_minus] = np.exp(-r * T) * np.maximum(S_minus[barrier_crossed_minus] - K, 0)
    V_plus[barrier_crossed_plus] = np.exp(-r * T) * np.maximum(S_plus[barrier_crossed_plus] - K, 0)

    V = (V_minus + V_plus) / 2

    aM = np.mean(V)
    bM = np.std(V, ddof=1)

    return aM, bM, S_plus, S_minus, V_minus, V_plus

aM_AT, bM_AT, S_plus, S_minus, V_minus, V_plus = down_and_out_milstein_Antithetic(1.5, 1000, 0.01, 0.1, 100, 90, 61, 10000)
correlation_stock_prices = np.corrcoef(S_plus, S_minus)[0, 1]
correlation_option_prices = np.corrcoef(V_plus, V_minus)[0, 1]

print(f"The correlation between final stock prices (S_plus and S_minus) is: {correlation_stock_prices}")
print(f"The correlation between final option prices (S_plus and S_minus) is: {correlation_option_prices}")

margin_of_error_AT = z_score * bM_AT / np.sqrt(10000)

lower_bound_AT = aM_AT - margin_of_error_AT
upper_bound_AT = aM_AT + margin_of_error_AT
width_CI_AT = upper_bound_AT - lower_bound_AT

acc_M_AT = (2* z_score * 100 * bM_AT)**2

comp_times_MS10000_AT = []
for i in range(10):
    start_time_MS_AT = time.perf_counter()
    down_and_out_milstein_Antithetic(1.5, 1000, 0.01, 0.1, 100, 90, 61, 10000)
    end_time_MS_AT = time.perf_counter()
    comp_times_AT = end_time_MS_AT - start_time_MS_AT
    comp_times_MS10000_AT.append(comp_times_AT)

compTimeMLSTN_AT_001 = (acc_M_AT / 10000) * np.mean(comp_times_MS10000_AT)

print(f"The down and out option price using the Milstein Scheme with Antithetic variates is {aM_AT}")
print(f"The confidence Interval using the Milstein Scheme with Antithetic variates is ({confidence_level * 100}%): [{lower_bound_AT}, {upper_bound_AT}]")
print(f"The width of the confidence interval using the Milstein Scheme with Antithetic variates is {width_CI_AT}")
print(f"The standard deviation from using the Milstein Scheme with Antithetic variates is {bM_AT}")
print(f"The mean compuational time using antithetic variates for 10000 Monte-Carlo simulations is {np.mean(comp_times_MS10000_AT)}")
print(f"The number of Monte-Carlo simulations required to have a CI width of 0.01 using antithetic variates is {acc_M_AT}")
print(f"The computational time required to achieve a CI = $0.01 using antithetic variates is {compTimeMLSTN_AT_001}")


# Question 3:
theta_opt = np.cov(V_MLSTN, S_MLSTN)[0, 1] / np.var(S_MLSTN)

def control_variate_estimate(V, S, theta, S0, r, T):
    E_S = S0 * np.exp(r * T)
    Z_theta = V + theta * (E_S - S)

    mean_Z_theta = np.mean(Z_theta)
    std_Z_theta = np.std(Z_theta, ddof=1)

    return mean_Z_theta, std_Z_theta, Z_theta

mean_Z_theta, std_Z_theta, Z_theta = control_variate_estimate(V_MLSTN, S_MLSTN, theta_opt, 100, 0.01, 3/2)

margin_of_error_CTRLVR = z_score * std_Z_theta / np.sqrt(10000)

lower_bound_CTRLVR = mean_Z_theta - margin_of_error_CTRLVR
upper_bound_CTRLVR = mean_Z_theta + margin_of_error_CTRLVR
width_CI_CTRLVR = upper_bound_CTRLVR - lower_bound_CTRLVR

acc_M_CTRLVR = (2* z_score * 100 * std_Z_theta)**2

comp_times_MS10000_COV = []
comp_times_MS10000_CTRLVR = []
for i in range(10):

    start_time_COV = time.perf_counter()
    np.cov(V_MLSTN, S_MLSTN)[0, 1] / np.var(S_MLSTN)
    end_time_COV = time.perf_counter()

    start_time_MS_CTRLVR = time.perf_counter()
    control_variate_estimate(V_MLSTN, S_MLSTN, theta_opt, 100, 0.01, 3/2)
    end_time_MS_CTRLVR = time.perf_counter()

    comp_times_COV = end_time_COV - start_time_COV
    comp_times_MS10000_COV.append(comp_times_COV)

    comp_times_CTRLVR = end_time_MS_CTRLVR - start_time_MS_CTRLVR
    comp_times_MS10000_CTRLVR.append(comp_times_CTRLVR)

CTRLVR_total_time  = np.mean(comp_times_MS10000_CTRLVR) + np.mean(comp_times_MS10000_COV) + np.mean(comp_times_MS10000)
compTimeMLSTN_CTRLVR_001 = (acc_M_CTRLVR / 10000) * CTRLVR_total_time

print(f"The down and out option price calculated using the control variate estimate is {mean_Z_theta}")
print(f"The confidence Interval using the Milstein Scheme with control variates is ({confidence_level * 100}%): [{lower_bound_CTRLVR}, {upper_bound_CTRLVR}]")
print(f"The width of the confidence interval using the Milstein Scheme with Antithetic variates is {width_CI_CTRLVR}")
print(f"The standard deviation, from using the control variate estimate is {std_Z_theta}")
print(f"The mean compuational time using control variates for 10000 Monte-Carlo simulations is {CTRLVR_total_time}")
print(f"The number of Monte-Carlo simulations required to have a CI width of 0.01 using control variates is {acc_M_CTRLVR}")
print(f"The computational time required to achieve a CI = $0.01 using control variates is {compTimeMLSTN_CTRLVR_001}")

theta_vals = np.linspace(-1, 3, 100)
var_Z_theta = []
for i in theta_vals:
    Z_theta = (control_variate_estimate(V_MLSTN, S_MLSTN, i, 100, 0.01, 3/2)[1])**2
    var_Z_theta.append(Z_theta)

plt.plot(theta_vals, var_Z_theta, label="Var(Z_theta)")
plt.axvline(theta_opt, color='red', linestyle='--', label=f"Optimal theta = {theta_opt:.2f}")
plt.xlabel("Theta")
plt.ylabel("Variance of Z_theta")
plt.title("Variance of Control Variate Estimator vs Theta")
plt.legend()
plt.grid()
plt.show()


# Question 4:
# Firstly, we will do the control variate first then antithetic
theta_plus = np.cov(V_plus, S_plus)[0, 1] / np.var(S_plus)
theta_minus = np.cov(V_minus, S_minus)[0, 1] / np.var(S_minus)

mean_Z_theta_plus, std_Z_theta_plus, Z_theta_plus = control_variate_estimate(V_plus, S_plus, theta_plus, 100, 0.01, 3/2)
mean_Z_theta_minus, std_Z_theta_minus, Z_theta_minus = control_variate_estimate(V_minus, S_minus, theta_minus, 100, 0.01, 3/2)

V_CTRLVR_AT = (Z_theta_plus + Z_theta_minus) / 2

std_CTRLVR_AT = np.std(V_CTRLVR_AT)
margin_of_error_CTRLVR_AT = z_score * std_CTRLVR_AT / np.sqrt(10000)

lower_bound_CTRLVR_AT = np.mean(V_CTRLVR_AT) - margin_of_error_CTRLVR_AT
upper_bound_CTRLVR_AT = np.mean(V_CTRLVR_AT) + margin_of_error_CTRLVR_AT
width_CI_CTRLVR_AT = upper_bound_CTRLVR_AT - lower_bound_CTRLVR_AT

acc_M_CTRLVR_AT = (2* z_score * 100 * std_CTRLVR_AT)**2

comp_times_CTRLVR_AT = np.mean(comp_times_MS10000_AT) + np.mean(comp_times_MS10000_CTRLVR) + np.mean(comp_times_MS10000_COV)
compTimeCTRLVR_AT_001 = (acc_M_CTRLVR_AT / 10000) * np.mean(comp_times_CTRLVR_AT)

print(f"The down and out option price calculated using the control variate then antithetic estimate is {np.mean(V_CTRLVR_AT)}")
print(f"The confidence Interval using the control variate then the antithetic estimate is ({confidence_level * 100}%): [{lower_bound_CTRLVR_AT}, {upper_bound_CTRLVR_AT}]")
print(f"The width of the confidence interval using the control variate then antithetic estimate is {width_CI_CTRLVR_AT}")
print(f"The standard deviation, from using the control variate then the antithetic estimate is {std_CTRLVR_AT}")
print(f"The mean compuational time using the antithetic then control variates for 10000 Monte-Carlo simulations is {comp_times_CTRLVR_AT}")
print(f"The number of Monte-Carlo simulations required to have a CI width of 0.01 using the control variate then antithetic estimate is {acc_M_CTRLVR_AT}")
print(f"The computational time required to achieve a CI = $0.01 using the control variate then antithetic estimate is {compTimeCTRLVR_AT_001}")

# Now doing the antithetic first, then control variate after:
V_AT = (V_plus + V_minus) / 2
S_AT = (S_plus + S_minus) / 2

theta_AT = np.cov(V_AT, S_AT)[0, 1] / np.var(S_AT)
mean_Z_theta_AT_CTRL, std_Z_theta_AT_CTRL, Z_theta_AT_CTRL = control_variate_estimate(V_AT, S_AT, theta_AT, 100, 0.01, 3/2)

margin_of_error_AT_CTRLVR = z_score * std_Z_theta_AT_CTRL / np.sqrt(10000)

lower_bound_AT_CTRLVR = np.mean(Z_theta_AT_CTRL) - margin_of_error_AT_CTRLVR
upper_bound_AT_CTRLVR = np.mean(Z_theta_AT_CTRL) + margin_of_error_AT_CTRLVR
width_CI_AT_CTRLVR = upper_bound_AT_CTRLVR - lower_bound_AT_CTRLVR

acc_M_AT_CTRLVR = (2* z_score * 100 * std_Z_theta_AT_CTRL)**2
comp_times_AT_CTRLVR = np.mean(comp_times_MS10000_AT) + np.mean(comp_times_MS10000_CTRLVR) + np.mean(comp_times_MS10000_COV)
compTimeAT_CTRLVR_001 = (acc_M_AT_CTRLVR / 10000) * np.mean(comp_times_AT_CTRLVR)

print(f"The down and out option price calculated using the antithetic then control variate estimate is {np.mean(Z_theta_AT_CTRL)}")
print(f"The confidence Interval using the antithetic then control variates is ({confidence_level * 100}%): [{lower_bound_AT_CTRLVR}, {upper_bound_AT_CTRLVR}]")
print(f"The width of the confidence interval using the antithetic then control variates is {width_CI_AT_CTRLVR}")
print(f"The standard deviation, from using the antithetic then control variates estimate is {std_Z_theta_AT_CTRL}")
print(f"The number of Monte-Carlo simulations required to have a CI width of 0.01 using the antithetic then control variates is {acc_M_AT_CTRLVR}")
print(f"The computational time required to achieve a CI = $0.01 using the antithetic then control variates is {compTimeAT_CTRLVR_001}")

aM_AT_acc, bM_AT_acc, S_plus_acc, S_minus_acc, V_minus_acc, V_plus_acc = down_and_out_milstein_Antithetic(1.5, 1000, 0.01, 0.1, 100, 90, 61, int(acc_M_AT_CTRLVR))
V_AT_acc = (V_plus_acc + V_minus_acc) / 2
S_AT_acc = (S_plus_acc + S_minus_acc) / 2

theta_AT_acc = np.cov(V_AT_acc, S_AT_acc)[0, 1] / np.var(S_AT_acc)
mean_Z_theta_AT_CTRL_acc, std_Z_theta_AT_CTRL_acc, Z_theta_AT_CTRL_acc = control_variate_estimate(V_AT_acc, S_AT_acc, theta_AT_acc, 100, 0.01, 3/2)
print(f"The accurate down and out put option price is {mean_Z_theta_AT_CTRL_acc}")
