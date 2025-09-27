import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.sparse import diags
from matplotlib import cm


# Question 1:
def tridiagonal_solve(lambda_val, b):
    n = len(b)
    diagonal = 1 + 2 * lambda_val
    off_diagonal = -lambda_val

    l, u = 1, 1
    ab = np.zeros((l + u + 1, n))

    ab[0, 1:] = off_diagonal
    ab[1, :] = diagonal
    ab[2, :-1] = off_diagonal

    x = solve_banded((l, u), ab, b)

    return x

b_implicit = np.array([4, 6, 8])
test_run = tridiagonal_solve(1, b_implicit)
print(test_run)

# Question 3:
def bounded_power_implicit(r, sigma, K, T, L, p, M, N, xmin, xmax):

    T_years = T / 12
    delta_tau = (sigma**2 * T_years) / (2 * N)
    delta_x = (xmax - xmin) / M
    lambda_val = delta_tau / (delta_x**2)

    w = np.zeros((M + 1, N + 1))
    x = np.linspace(xmin, xmax, M + 1)
    tau = np.linspace(0, (sigma**2 * T_years)/2, N + 1) # From t = T to t = 0
    
    # Tau = 0 part:
    for i in range(M + 1):
        S = np.exp(x[i])
        vanilla_put = max(K - S, 0)
        if p == 1: # When p = 1 and the boundary is infinite we treat the option like a vanilla European put
            if L == float('inf'):
                w[i, 0] = vanilla_put
        if p != 1:
            powered_put = vanilla_put ** p
            w[i, 0] = min(L, powered_put)
    
    for i in range(N):
        t = T_years - (2 * tau[i + 1]) / sigma**2
        
        # Boundary conditions for when S -> 0 and S -> inf
        if p == 1:
            if L == float('inf'):
                w[0, i + 1] = K * np.exp(-r * t)
                w[M, i + 1] = 0
        if p != 1:
            w[0, i + 1] = min(L, (K ** p)) * np.exp(-r * t)
            w[M, i + 1] = 0
        
        # Setting the right hand side of the equation up
        b = np.array(w[0:M+1, i])
        b[0] += lambda_val * w[0, i + 1]
        b[-1] += lambda_val * w[M, i + 1]
        w[0:M+1, i + 1] = tridiagonal_solve(lambda_val, b)
    
    t_adj = (2 * r / sigma**2) - 1
    S_grid = np.zeros((M + 1, N + 1))
    t_grid = np.zeros(N + 1)
    for j in range(N + 1):
        t_grid[j] = T_years - (2 * tau[j]) / sigma**2
        for i in range(M + 1):
            S_grid[i, j] = np.exp(x[i] - t_adj * tau[j])
    
    return w, S_grid, t_grid, b, lambda_val

r = 0.08
sigma = 0.1
K = 10
T = 18
L_europ = float('inf')
p_europ = 1
M = 1000
N = 1000
xmin = -10
xmax = 3
w_europ, s_grid_europ, t_grid_europ, b_europ, lambda_europ = bounded_power_implicit(r, sigma, K, T, L_europ, p_europ, M, N, xmin, xmax)

def black_scholes_put(S_0, r, sigma, T, K):

    T_years = T / 12
    d_1 = ((np.log(S_0/K)) + (r + (sigma**2)/2) * T_years) / (sigma * np.sqrt(T_years))
    d_2 = d_1 - (sigma * np.sqrt(T_years))
    price_of_put = K * np.exp(-r * T_years) * norm.cdf(-d_2) - S_0 * norm.cdf(-d_1)

    return price_of_put

S0_european = s_grid_europ[:, -1]
option_price_european = w_europ[:, -1]

true_europ_put_opt_prices = []
for i in S0_european:
    price = black_scholes_put(i, r, sigma, T, K)
    true_europ_put_opt_prices.append(price)

plt.plot(S0_european, option_price_european, 'b-', label = 'Numerical (Finite Difference)')
plt.plot(S0_european, true_europ_put_opt_prices, 'r--', label = 'Analytical (Black-Scholes)')
plt.xlabel('Stock Price (S)')
plt.ylabel('Option Price')
plt.title('Vanilla European Put: Numerical vs Analytical Solution at t=0')
plt.xlim(0, 20)
plt.legend()
plt.grid(True)
plt.show()

def tridiagonal_matrix(lambda_val, n):
    main_diag = np.ones(n + 1) * (1 + 2 * lambda_val)
    off_diag = np.ones(n) * (-lambda_val)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n + 1, n + 1)).toarray()
    return A

def errors(w, A, b):
    A_inv = np.linalg.inv(A)
    error = w - (A_inv @ b)
    return error

A_europ = tridiagonal_matrix(lambda_europ, M)
err = errors(option_price_european, A_europ, b_europ)

A_500 = tridiagonal_matrix(lambda_europ, 500)
w_europ500, s_grid_europ500, t_grid_europ500, b_europ500, lambda_europ500 = bounded_power_implicit(r, sigma, K, T, L_europ, p_europ, 500, N, xmin, xmax)
err100 = errors(w_europ500[:, -1], A_500, b_europ500)

A_750 = tridiagonal_matrix(lambda_europ, 750)
w_europ750, s_grid_europ750, t_grid_europ750, b_europ750, lambda_europ750 = bounded_power_implicit(r, sigma, K, T, L_europ, p_europ, 750, N, xmin, xmax)
err500 = errors(w_europ750[:, -1], A_750, b_europ750)

A_900 = tridiagonal_matrix(lambda_europ, 900)
w_europ900, s_grid_europ900, t_grid_europ900, b_europ900, lambda_europ900 = bounded_power_implicit(r, sigma, K, T, L_europ, p_europ, 900, N, xmin, xmax)
err900 = errors(w_europ900[:, -1], A_900, b_europ900)

plt.plot(s_grid_europ500[:, 1], err100, 'b', label = 'M = 500')
plt.plot(s_grid_europ750[:, 1], err500, 'r', label = 'M = 750')
plt.plot(s_grid_europ900[:, 1], err900, 'g', label = 'M = 900')
plt.xlabel('Stock Price (S)')
plt.ylabel('Round-Off Error')
plt.title('The Round-Off Error for the Implicit Scheme Against Stock Price')
plt.ylim(-0.00075, 0.0002)
plt.xlim(0.05, 20)
plt.legend()
plt.grid(True)
plt.show()

# Question 4:
L2 = 81
p2 = 2
# The grid parameters generated for this part used a lower spatial resolution for quicker computation
w2, s2, t2, b2, lambda2 = bounded_power_implicit(r, sigma, K, T, L2, p2, 100, N, xmin, xmax)
t2_grid, s2_grid = np.meshgrid(t2, s2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(t2, s2, w2, cmap=cm.viridis, linewidth=0, antialiased=True)
plt.ylim(0, 20)

ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.25, aspect=5)

ax.set_xlabel('Time to Maturity (years)')
ax.set_ylabel('Stock Price S')
ax.set_zlabel('Option Price V')
plt.tight_layout()
plt.show()

# Now plotting current value vs current payoff:
# We create a new grid with higher time resolution for this part and part 5
w_theta1, s_theta1, t_theta1, b_theta1, lambda_theta1 = bounded_power_implicit(r, sigma, K, T, L2, p2, M, N, xmin, xmax)
w2_current = w_theta1[:, -1]
s2_current = np.array(s_theta1[:, -1])

def power_put_opt_payoff(L, p, k, s):
    power_opt = (max(0, (k - s))) ** p
    return min(L, power_opt)

payoffs1 = []
for i in s2_current:
    payoff = power_put_opt_payoff(L2, p2, K, i)
    payoffs1.append(payoff)

plt.plot(s2_current, np.array(payoffs1), 'b-', label = 'Current Option Payoff (t = 0)')
plt.plot(s2_current, w2_current, 'r--', label = 'Current Option Value (t = 0)')
plt.xlabel('Stock Price (S)')
plt.ylabel('Option Price')
plt.title('Current Option Value vs Payoff for p = 2')
plt.xlim(0, 20)
plt.legend()
plt.grid(True)
plt.show()

target_S_val = 4.73
estimated_price1 = np.interp(target_S_val, s2_current, w2_current)
consistent_S_prices = np.linspace(0, 18, 1001)


V1_t0 = np.interp(consistent_S_prices, s2_current, w2_current)
V1_t1 = np.interp(consistent_S_prices, s_theta1[:, -2], w_theta1[:, -2])

delta_t = t_theta1[0] - t_theta1[1] # We will use the same delta t for both p = 2 and p = 0.5

Theta1 = (V1_t1 - V1_t0) / delta_t
max_theta_idx1 = np.argmax(Theta1)
S_max_theta1 = consistent_S_prices[max_theta_idx1]
max_theta1 = Theta1[max_theta_idx1]

print(f"Estimated price at S = {target_S_val}: €{estimated_price1:.2f}")
print(f"Stock price with largest Theta: €{S_max_theta1:.2f}")
print(f"Maximum Theta value: {max_theta1:.2f}")

plt.plot(consistent_S_prices[1:], Theta1[1:], 'b-', label='FD Theta (Forward Difference)')
plt.axvline(S_max_theta1, color='r', linestyle='--', label='Max Theta')
plt.xlabel('Stock Price (S)')
plt.ylabel('Theta')
plt.title('Theta Comparison: Finite Difference vs Analytical (Vanilla Put)')
plt.grid(True)
plt.legend()
plt.show()


# Question 5:
L3 = 3
p3 = 0.5
w3, s3, t3, b3, lambda3 = bounded_power_implicit(r, sigma, K, T, L3, p3, 100, N, xmin, xmax)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(t3, s3, w3, cmap=cm.viridis, linewidth=0, antialiased=True)

plt.ylim(0, 20)

ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.25, aspect=5)

ax.set_xlabel('Time to Maturity (years)')
ax.set_ylabel('Stock Price S')
ax.set_zlabel('Option Price V')
plt.tight_layout()
plt.show()

w_theta2, s_theta2, t_theta2, b_theta2, lambda_theta2 = bounded_power_implicit(r, sigma, K, T, L3, p3, M, N, xmin, xmax)
w3_current = w_theta2[:, -1]
s3_current = np.array(s_theta2[:, -1])

payoffs2 = []
for i in s3_current:
    payoff = power_put_opt_payoff(L3, p3, K, i)
    payoffs2.append(payoff)

plt.plot(s3_current, np.array(payoffs2), 'b-', label = 'Current Option Payoff (t = 0)')
plt.plot(s3_current, w3_current, 'r--', label = 'Current Option Value (t = 0)')
plt.xlabel('Stock Price (S)')
plt.ylabel('Option Price')
plt.title('Current Option Value vs Payoff for p = 0.5')
plt.xlim(0, 20)
plt.legend()
plt.grid(True)
plt.show()

estimated_price2 = np.interp(target_S_val, s3_current, w3_current)

V2_t0 = np.interp(consistent_S_prices, s3_current, w3_current)
V2_t1 = np.interp(consistent_S_prices, s_theta2[:, -2], w_theta2[:, -2])

Theta2 = (V2_t1 - V2_t0) / delta_t
max_theta_idx2 = np.argmax(Theta2)
S_max_theta2 = consistent_S_prices[max_theta_idx2]
max_theta2 = Theta2[max_theta_idx2]

print(f"Estimated price at S = {target_S_val}: €{estimated_price2:.2f}")
print(f"Stock price with largest Theta: €{S_max_theta2:.2f}")
print(f"Maximum Theta value: {max_theta2:.2f}")