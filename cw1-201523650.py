import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gammaln
from scipy.stats import norm
import itertools
import pandas as pd

# Question 1:
def main():

    S_0 = input("Enter the first parameter: ")
    r = input("Enter the second parameter: ")
    sigma = input("Enter the third parameter: ")
    T = input("Enter the Fourth parameter: ")
    K = input("Enter the Fifth parameter: ")
    M = input("Enter the Sixth parameter: ")

    error_occurred = False
    try:
        if S_0.isalpha():
            raise TypeError("Error: Stock price must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if float(S_0) < 0:
            raise ValueError("Error: Stock price must not be negative")
    except ValueError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if r.isalpha():
            raise TypeError("Error: interest rate must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if float(r) < 0:
            raise ValueError("Error: interest rate must not be negative")
    except ValueError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if sigma.isalpha():
            raise TypeError("Error: Volatility must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if float(sigma) < 0:
            raise ValueError("Error: volatility must not be negative")
    except ValueError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if T.isalpha():
            raise TypeError("Error: Time must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        error_occurred = True
        
    try:
        if float(T) < 0:
            raise ValueError("Error: Time must not be negative")
    except ValueError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if K.isalpha():
            raise TypeError("Error: Strike price must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if float(K) < 0:
            raise ValueError("Error: Strike price must not be negative")
    except ValueError as e:
        print(f"{e}")
        error_occurred = True

    try:
        if M.isalpha():
            raise TypeError("Error: Number of periods must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        error_occurred = True

    try: 
        if float(M) <= 0:
            raise ValueError("Error: Number of periods must be greater than zero")
        if int(float(M)) != float(M):
            raise ValueError("Error: M must be an integer")
                       
    except ValueError as e:
        print(f"{e}")
        error_occurred = True
    
    if error_occurred:
        print("Use valid inputs")
    else:
        return S_0, r, sigma, T, K, M

result = main()

if result:
    S_0, r, sigma, T, K, M = result
    print(f"S_0 = {S_0}, r = {r}, sigma = {sigma}, T = {T}, K = {K}, M = {M}")

# Question 2:

# Firstly we will implement the code for the
# inefficient original call and put option pricing


def binomial_call(S_0, r, sigma, T, K, M):

    delta_t = T / M
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - d) / (u - d)

    if p <= 0 or p >= 1:
        print('Error - not arbitrage free')
    
    V=np.zeros((M + 1, M + 1))
    S = np.zeros((M + 1, M + 1))

    for j in range(M+1):
        
        S[j, M] = S_0 * (u ** j) * (d ** (M - j))
        V[j,M] = np.maximum(S[j, M] - K, 0)
        
    for i in range(M-1, -1, -1):
        for j in range(i+1):
            S[j, i] = S_0 * (u ** j) * (d**(i - j))
            V[j, i] = np.exp(-r * delta_t) * (p * V[j + 1, i + 1] + (1 - p) * V[j, i + 1])
    
    i = np.arange(M + 1)
    log_binomial = gammaln(M + 1) - gammaln(i + 1) - gammaln(M - i + 1)
    log_prob = i * np.log(p) + (M - i) * np.log(1 - p)
    combo_prob = np.exp(log_binomial + log_prob)
    
    Mean = sum(S[:, M] * combo_prob)
    Second_Moment = sum((S[:, M] ** 2) * combo_prob)

    return [V[0, 0], Mean, Second_Moment]


def binomial_put(S_0, r, sigma, T, K, M):

    delta_t = T / M
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - d) / (u - d)

    if p <= 0 or p >= 1:
        print('Error - not arbitrage free')
    
    V=np.zeros((M + 1, M + 1))
    S = np.zeros((M + 1, M + 1))

    for j in range(M+1):
        
        S[j, M] = S_0 * (u ** j) * (d ** (M - j))
        V[j,M] = np.maximum(K - S[j, M], 0)
        
    for i in range(M-1, -1, -1):
        for j in range(i+1):
            S[j, i] = S_0 * (u ** j) * (d**(i - j))
            V[j, i] = np.exp(-r * delta_t) * (p * V[j + 1, i + 1] + (1 - p) * V[j, i + 1])
    
    i = np.arange(M + 1)
    log_binomial = gammaln(M + 1) - gammaln(i + 1) - gammaln(M - i + 1)
    log_prob = i * np.log(p) + (M - i) * np.log(1 - p)
    combo_prob = np.exp(log_binomial + log_prob)
    
    Mean = sum(S[:, M] * combo_prob)
    Second_Moment = sum((S[:, M] ** 2) * combo_prob)

    return [V[0, 0], Mean, Second_Moment]

# Now we will implement the improved version

def binomial_CRR(S_0, r, sigma, T, K, M, call_option, u_input, exact_CRR, tilted_tree):

    # These are errors to catch invalid inputs for our function (question 5)
    try:
        if u_input == True:
            if exact_CRR == True:
                if tilted_tree ==  True:
                    raise ValueError("Excluding call_option, only one of the boolean arguments can be True.")
    except ValueError as e:
            print(f"{e}")
            return None

    try:
        if u_input == True:
            if exact_CRR == True:
                if tilted_tree ==  False:
                    raise ValueError("Excluding call_option, only one of the boolean arguments can be True.")
    except ValueError as e:
            print(f"{e}")
            return None

    try:
        if u_input == True:
            if exact_CRR == False:
                if tilted_tree ==  True:
                    raise ValueError("Excluding call_option, only one of the boolean arguments can be True.")
    except ValueError as e:
            print(f"{e}")
            return None

    try:
        if u_input == False:
            if exact_CRR == True:
                if tilted_tree ==  True:
                    raise ValueError("Excluding call_option, only one of the boolean arguments can be True.")
    except ValueError as e:
            print(f"{e}")
            return None

    # u_input for the function was implemented for question 4
    if u_input == False:

        delta_t = T / M

        # This part was implemented for question 5

        if tilted_tree == True:

            x = (1 / M) * np.log(K / S_0)
            u = np.exp(x + (sigma * np.sqrt(delta_t)))
            d = np.exp(x - (sigma * np.sqrt(delta_t)))

        if exact_CRR == True:

            beta = 0.5 * (np.exp(-r * delta_t) + np.exp((r + (sigma ** 2)) * delta_t))

            u = beta + np.sqrt((beta ** 2) - 1)
            d = 1 / u

        if exact_CRR == False: 
            if tilted_tree == False:

                u = np.exp(sigma * np.sqrt(delta_t))
                d = np.exp(-sigma * np.sqrt(delta_t))

        p = (np.exp(r * delta_t) - d) / (u - d)

        try:
            if p <= 0 or p >= 1:
                raise ValueError("Error: Not arbitrage free")
        except ValueError as e:
            print(f"{e}")
            return None

    u_vals = []
    variable_opt_u = "optimised_u_val"

    if u_input == True:
        delta_t = T / M

        u_start = input("Enter the start u value: ")

        if u_start == variable_opt_u:
            u_start = optimised_u_val

        u_start = float(u_start)
        d_start = 1 / u_start

        decision = input("Do you want to input an end value? yes/no ")
        if decision == "yes":
            u_end = input("Enter the end u value: ")
            u_end = float(u_end)

            d_end = 1/ u_end
        
            try:
                if d_end < u_end < np.exp(r * delta_t):
                    raise ValueError("Error: Not arbitrage free")
                if d_end > u_end > np.exp(r * delta_t):
                    raise ValueError("Error: Not arbitrage free")
            except ValueError as e:
                print(f"{e}")
                return None
           
            u = np.linspace(u_start, u_end, 20000)
            d = np.linspace(d_start, d_end, 20000)
            u_vals = u

        if decision == "no":
            u = u_start
            d = d_start

        p = (np.exp(r * delta_t) - d) / (u - d)

        try:
            if d_start < u_start < np.exp(r * delta_t):
                raise ValueError("Error: Not arbitrage free")
            if d_start > u_start > np.exp(r * delta_t):
                raise ValueError("Error: Not arbitrage free")
        except ValueError as e:
            print(f"{e}")
            return None
    
    try:
        if str(S_0).isalpha():
            raise TypeError("Error: Stock price must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        return None

    try:
        if float(S_0) < 0:
            raise ValueError("Error: Stock price must not be negative")
    except ValueError as e:
        print(f"{e}")
        return None
    
    try:
        if str(r).isalpha():
            raise TypeError("Error: interest rate must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        return None
    
    try:
        if float(r) < 0:
            raise ValueError("Error: interest rate must not be negative")
    except ValueError as e:
        print(f"{e}")
        return None
    
    try:
        if str(sigma).isalpha():
            raise TypeError("Error: Volatility must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        return None
    
    try:
        if float(sigma) < 0:
            raise ValueError("Error: volatility must not be negative")
    except ValueError as e:
        print(f"{e}")
        return None
    
    try:
        if str(T).isalpha():
            raise TypeError("Error: Time must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        return None
    
    try:
        if float(T) < 0:
            raise ValueError("Error: Time must not be negative")
    except ValueError as e:
        print(f"{e}")
        return None
    
    try:
        if str(K).isalpha():
            raise TypeError("Error: Strike price must have a numerical value")
    except TypeError as e:
        print(f"{e}")
        return None
    
    try:
        if float(K) < 0:
            raise ValueError("Error: Strike price must not be negative")
    except ValueError as e:
        print(f"{e}")
        return None
    
    try:
        if str(M).isalpha():
            raise TypeError("Error: Number of periods must have a numerical value")
    except TypeError as e:
        print(f"{e}")

    try:
        if float(M) <= 0:
            raise ValueError("Error: Number of periods must be greater than zero")
        if int(float(M)) != float(M):
            raise ValueError("Error: M must be an integer")
    except ValueError as e:
        print(f"{e}")
        return None
    
    try:
        if type(call_option) != bool:
            raise TypeError("Error: Option type must be a boolean value")
    except TypeError as e:
        print(f"{e}")
        return None

    PV = 0
    Mean = 0
    Second_Moment = 0
    for i in range(M + 1):

        log_binomial = gammaln(M + 1) - gammaln(i + 1) - gammaln(M - i + 1)
        log_prob = i * np.log(p) + (M - i) * np.log(1 - p)
        combo_prob = np.exp(log_binomial + log_prob)
        discounting_factor = np.exp(-r * T)
        St = S_0 * (u**i) * (d**(M - i))

        if call_option == True:
            V = np.maximum(St - K, 0)
        else:
            V = np.maximum(K - St, 0)
        
        PV += combo_prob * discounting_factor * V
        Mean += combo_prob * St
        Second_Moment += combo_prob * (St ** 2)
    
    if len(u_vals) != 0:
        return [PV, Mean, Second_Moment, u_vals]

    return [PV, Mean, Second_Moment]

print(f"The call option price from the binomial model under the CRR approximation is {binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, True, False, False, False)[0]}")
print(f"The put option price from the binomial model under the CRR approximation is {binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, False, False, False, False)[0]}")

M_vals = np.arange(20, 500, 20)

comp_times_oc = []
for i in M_vals:
    start_time_oc = time.perf_counter()
    result = binomial_CRR(285, 0.021, 0.12, 5/3, 290, i, True, False, False, False)
    end_time_oc = time.perf_counter()
    time_optimised_call = end_time_oc - start_time_oc
    comp_times_oc.append(time_optimised_call)

comp_times_op = []
for i in M_vals:
    start_time_op = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, i, False, False, False, False)
    end_time_op = time.perf_counter()
    time_optimised_put = end_time_op - start_time_op
    comp_times_op.append(time_optimised_put)

comp_times_p = []
for i in M_vals:
    start_time_p = time.perf_counter()
    binomial_put(285, 0.021, 0.12, 5/3, 290, i)
    end_time_p = time.perf_counter()
    time_put = end_time_p - start_time_p
    comp_times_p.append(time_put)

comp_times_c = []
for i in M_vals:
    start_time_c = time.perf_counter()
    binomial_call(285, 0.021, 0.12, 5/3, 290, i)
    end_time_c = time.perf_counter()
    time_call = end_time_c - start_time_c
    comp_times_c.append(time_call)

plt.loglog(M_vals, comp_times_c, marker = 'o', color='b', label='Computational time of inefficient call function')
plt.loglog(M_vals, comp_times_oc, marker = 'o', color='r', label='Computational time of efficient call function')
plt.title("Computational Time of 2 Binomial Call Pricing Functions")
plt.xlabel("Number of Time Steps")
plt.ylabel("Computational Time")
plt.grid(True)
plt.legend()
plt.show()

plt.loglog(M_vals, comp_times_p, marker = 'o', color='b', label='Computational time of inefficient put function')
plt.loglog(M_vals, comp_times_op,  marker = 'o', color='r', label='Computational time of efficient put function')
plt.title("Computational Time of 2 Binomial Put Pricing Functions")
plt.xlabel("Number of Time Steps")
plt.ylabel("Computational Time")
plt.grid(True)
plt.legend()
plt.show()

# Question 3:
# Log-Normal mean and variance was calculated later for Q4

def exact_price_call(S_0, r, sigma, T, K):

    d_1 = ((np.log(S_0/K)) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - (sigma * np.sqrt(T))

    price_of_call = S_0 * stats.norm.cdf(d_1) - K * np.exp(-r * T) * stats.norm.cdf(d_2)

    mu = S_0 * np.exp(r * T)
    second_moment = (S_0 ** 2)  * np.exp(((2 * r ) + (sigma ** 2)) * T)

    return price_of_call, mu, second_moment

def exact_price_put(S_0, r, sigma, T, K):

    d_1 = ((np.log(S_0/K)) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - (sigma * np.sqrt(T))

    price_of_put = K * np.exp(-r * T) * stats.norm.cdf(-d_2) - S_0 * stats.norm.cdf(-d_1)

    return price_of_put

Black_Scholes_Call_Option_Price = exact_price_call(285, 0.021, 0.12, 5/3, 290)[0]
Black_Scholes_Put_Option_Price = exact_price_put(285, 0.021, 0.12, 5/3, 290)
Black_Scholes_Call_Mean = exact_price_call(285, 0.021, 0.12, 5/3, 290)[1]
Black_Scholes_Call_Second_Moment = exact_price_call(285, 0.021, 0.12, 5/3, 290)[2]

print(f"The exact Black-Scholes price for the call option is {Black_Scholes_Call_Option_Price}")
print(f"The exact Black-Scholes price for the put option is {Black_Scholes_Put_Option_Price}")
print(f"The call option price from the binomial model under the CRR approximation is {binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, True, False, False, False)[0]}")
print(f"The put option price from the binomial model under the CRR approximation is {binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, False, False, False, False)[0]}")

odd_M = np.arange(5, 500, 2)
even_M = np.arange(4, 500, 2)

accuracy_plot_odd = []
for i in odd_M:
    val_odd = binomial_CRR(285, 0.021, 0.12, 5/3, 290, i, True, False, False, False)[0]
    accuracy_plot_odd.append(val_odd)

accuracy_plot_even = []
for j in even_M:
    val_even = binomial_CRR(285, 0.021, 0.12, 5/3, 290, j, True, False, False, False)[0]
    accuracy_plot_even.append(val_even)

plt.plot(odd_M, np.array(accuracy_plot_odd), color = 'r', label = "M Odd")
plt.plot(even_M, np.array(accuracy_plot_even), color = 'b', label = "M Even")
plt.axhline(Black_Scholes_Call_Option_Price, color = 'grey', linestyle = '--', label = "Black-Scholes price")
plt.xlabel("Number of periods")
plt.ylabel("Option price")
plt.grid(True)
plt.title("Convergence of the Binomial Model")
plt.legend()
plt.show()

# Question 4:

print(f"The mean stock price from the binomial model under the CRR approximation is {binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, True, False, False, False)[1]}")
print(f"The second moment from the binomial model under the CRR approximation is {binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, False, False, False, False)[2]}")

def binomial_stock_dist(S_0, r, sigma, T, M):

    delta_t = T / M
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    prices = []
    probabilities = []

    for i in range(M + 1):

        St = S_0 * (u ** i) * (d ** (M - i))
        prices.append(St)

    for i in range(M + 1):

        log_binomial = gammaln(M + 1) - gammaln(i + 1) - gammaln(M - i + 1)
        log_prob = i * np.log(p) + (M - i) * np.log(1 - p)
        combo_prob = np.exp(log_binomial + log_prob)
        probabilities.append(combo_prob)
    
    return [np.array(prices), np.array(probabilities)]

def black_scholes_lognormal_pdf(S_0, r, sigma, T, s):

    mu_log = np.log(S_0) + (r - 0.5 * sigma**2) * T
    sigma_log = sigma * np.sqrt(T)
    
    denom = s * sigma_log * np.sqrt(2 * np.pi)
    exponent = -((np.log(s) - mu_log)**2) / (2 * sigma_log**2)

    pdf = np.exp(exponent) / denom
    
    return pdf

def black_scholes_lognormal_cdf(S_0, r, sigma, T, s):

    mu_log = np.log(S_0) + (r - 0.5 * sigma**2) * T
    sigma_log = sigma * np.sqrt(T)
    
    cdf = norm.cdf((np.log(s) - mu_log) / sigma_log)
    
    return cdf

s_values = np.linspace(0, 800, 300)
pdf_values = black_scholes_lognormal_pdf(285, 0.021, 0.12, 5/3, s_values)

bin_prices = binomial_stock_dist(285, 0.021, 0.12, 5/3, 300)[0]
bin_probs = binomial_stock_dist(285, 0.021, 0.12, 5/3, 300)[1]

plt.hist(bin_prices, weights = bin_probs, density = True, bins = 200, edgecolor = 'black', color = 'red', label = 'Binomial CDF')
plt.xlim(0, 800)
plt.plot(s_values, pdf_values, color = 'blue', label = 'BS CDF')
plt.xlabel('Stock Price at T')
plt.ylabel('Probability')
plt.grid(True)
plt.title('PDF of the binomial-modelled stock prices')
plt.show()

cdf_values = black_scholes_lognormal_cdf(285, 0.021, 0.12, 5/3, s_values)
cumulative_probs_binomial = np.array(list(itertools.accumulate(bin_probs)))

plt.hist(bin_prices, weights = cumulative_probs_binomial, bins = 1000, edgecolor = 'black', color = 'red', label = 'Binomial CDF')
plt.xlim(100, 500)
plt.plot(s_values, cdf_values, color = 'blue', label = 'BS CDF')
plt.xlabel('Stock Price at T')
plt.ylabel('Probability')
plt.grid(True)
plt.title('CDF of the binomial-modelled stock prices')
plt.show()

# Run this code before running the plotting code
# Testing from 1.02 to 1.025 shows visible difference b/w black scholes and approximate CRR
bin_crr = binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, True, True, False, False)
optionprice_u = bin_crr[0]
u_vals = bin_crr[3]

plt.plot(u_vals, optionprice_u, color = 'red', label = 'Binomial Model (varied u)')
plt.axhline(Black_Scholes_Call_Option_Price, color = 'grey', linestyle = '--', label = "Black-Scholes model")
plt.axhline(binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, True, False, False, False)[0], color = 'blue', label = 'Binomial Model (approximate CRR)')
plt.xlabel("u value")
plt.ylabel("Option Price")
plt.title('Option price as a function of u')
plt.grid()
plt.legend()
plt.show()

diffs = np.abs(optionprice_u - Black_Scholes_Call_Option_Price)
idx_min = np.argmin(diffs)
optimised_u_val = u_vals[idx_min]

# Enter optimised_u_val for M = 50 into the variables below
adjusted_binomial_50 = binomial_CRR(285, 0.021, 0.12, 5/3, 290, 50, True, True, False, False)
adjusted_bin_option_price_50 = adjusted_binomial_50[0]
adjusted_mean_stock_price_bin_50 = adjusted_binomial_50[1]
adjusted_second_moment_bin_50 = adjusted_binomial_50[2]

print(f"The value of u that matches closest to the black scholes model is: {optimised_u_val}")
print(f"The binomial price of this call option with the optimised u value {adjusted_bin_option_price_50}")
print(f"The exact black scholes call option price is {Black_Scholes_Call_Option_Price}")

print(f"The mean stock price from the binomial call function under the optimised u value is {adjusted_mean_stock_price_bin_50}")
print(f"The mean stock price calculated by the black scholes model is {Black_Scholes_Call_Mean}")

print(f"The second moment from the binomial call function under the optimised u value is {adjusted_second_moment_bin_50}")
print(f"The second moment calculated by the black scholes model is {Black_Scholes_Call_Second_Moment}")

# Question 5:
# We created a seperate function for the alternative u value to avoid inputting 'optimised_u_val' each time
def alt_u_CRR(S_0, r, T, K, M, call_option):
    delta_t = T / M
    u = optimised_u_val
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)

    PV = 0
    Mean = 0
    Second_Moment = 0
    for i in range(M + 1):

        log_binomial = gammaln(M + 1) - gammaln(i + 1) - gammaln(M - i + 1)
        log_prob = i * np.log(p) + (M - i) * np.log(1 - p)
        combo_prob = np.exp(log_binomial + log_prob)
        discounting_factor = np.exp(-r * T)
        St = S_0 * (u**i) * (d**(M - i))

        if call_option == True:
            V = np.maximum(St - K, 0)
        else:
            V = np.maximum(K - St, 0)
        
        PV += combo_prob * discounting_factor * V
        Mean += combo_prob * St
        Second_Moment += combo_prob * (St ** 2)
    
    if len(u_vals) != 0:
        return [PV, Mean, Second_Moment, u_vals]

    return [PV, Mean, Second_Moment]

iterations = 100
# Table values for 50 periods:
M1 = 50
# We re-define adjusted_binomial_50 again using the alt_u_CRR()
# function in case the user forgets to input the optimised_u_val
adjusted_binomial_50 = alt_u_CRR(285, 0.021, 5/3, 290, M1, True)
adjusted_bin_option_price_50 = adjusted_binomial_50[0]
adjusted_mean_stock_price_bin_50 = adjusted_binomial_50[1]
adjusted_second_moment_bin_50 = adjusted_binomial_50[2]

exact_crr_price_50 = binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, True, False)[0]
tilted_tree_price_50 = binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[0]
richardson_price_50 = (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[0]
                       - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M1 / 2), True, False, False, True)[0])


err_crr_approx_50 = (np.abs(binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, False)[0]
                             - Black_Scholes_Call_Option_Price))
err_exact_crr_50 = np.abs(exact_crr_price_50 - Black_Scholes_Call_Option_Price)
err_alt_u_50 = np.abs(adjusted_bin_option_price_50 - Black_Scholes_Call_Option_Price)
err_tilted_tree_50 = np.abs(tilted_tree_price_50 - Black_Scholes_Call_Option_Price)
err_richardson_price_50 = np.abs(richardson_price_50 - Black_Scholes_Call_Option_Price)

mean_richardson_50 = (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[1] 
                       - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M1 / 2), True, False, False, True)[1])
second_moment_richardson_50 = (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[2] 
                       - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M1 / 2), True, False, False, True)[2])

comp_time_BS = []
for i in range(iterations):
    start_time_BS = time.perf_counter()
    exact_price_call(285, 0.021, 0.12, 5/3, 290)[0]
    end_time_BS = time.perf_counter()
    BS_times = end_time_BS - start_time_BS
    comp_time_BS.append(BS_times)
comp_time_BS = sum(comp_time_BS) / iterations

comp_time_CRR_approx_50 = []
for i in range(iterations):
    start_time_CRR_approx_50 = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, False)[0]
    end_time_CRR_approx_50 = time.perf_counter()
    CRR_approx_times_50 = end_time_CRR_approx_50 - start_time_CRR_approx_50
    comp_time_CRR_approx_50.append(CRR_approx_times_50)
comp_time_CRR_approx_50 = sum(comp_time_CRR_approx_50) / iterations

comp_time_CRR_exact_50 = []
for i in range(iterations):
    start_time_CRR_exact_50 = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, True, False)[0]
    end_time_CRR_exact_50 = time.perf_counter()
    CRR_exact_times_50 = end_time_CRR_exact_50 - start_time_CRR_exact_50
    comp_time_CRR_exact_50.append(CRR_exact_times_50)
comp_time_CRR_exact_50 = sum(comp_time_CRR_exact_50) / iterations

comp_time_CRR_alt_50 = []
for i in range(iterations):
    start_time_CRR_alt_50 = time.perf_counter()
    alt_u_CRR(285, 0.021, 5/3, 290, M1, True)[0]
    end_time_CRR_alt_50 = time.perf_counter()
    alt_CRR_times_50 = end_time_CRR_alt_50 - start_time_CRR_alt_50
    comp_time_CRR_alt_50.append(alt_CRR_times_50)
comp_time_CRR_alt_50 = sum(comp_time_CRR_alt_50) / iterations

comp_time_tilted_tree_50 = []
for i in range(iterations):
    start_time_tilted_50 = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[0]
    end_time_tilted_50 = time.perf_counter()
    tilted_times_50 = end_time_tilted_50 - start_time_tilted_50
    comp_time_tilted_tree_50.append(tilted_times_50)
comp_time_tilted_tree_50 = sum(comp_time_tilted_tree_50) / iterations

comp_time_richardson_50 = []
for i in range(iterations):
    start_time_richardson_50 = time.perf_counter()
    (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[0]
                        - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M1 / 2), True, False, False, True)[0])
    end_time_richardson_50 = time.perf_counter()
    richardson_times_50 = end_time_richardson_50 - start_time_richardson_50
    comp_time_richardson_50.append(richardson_times_50)
comp_time_richardson_50 = sum(comp_time_richardson_50) / iterations


# Table Values for 48 periods:
M2 = 48
adjusted_binomial_48 = alt_u_CRR(285, 0.021, 5/3, 290, M2, True)
adjusted_bin_option_price_48 = adjusted_binomial_48[0]
adjusted_mean_stock_price_bin_48 = adjusted_binomial_48[1]
adjusted_second_moment_bin_48 = adjusted_binomial_48[2]

exact_crr_price_48 = binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, True, False)[0]
tilted_tree_price_48 = binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[0]
richardson_price_48 = (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[0]
                        - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M2 / 2), True, False, False, True)[0])

err_crr_approx_48 = (np.abs(binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, False)[0]
                             - Black_Scholes_Call_Option_Price))
err_exact_crr_48 = np.abs(exact_crr_price_48 - Black_Scholes_Call_Option_Price)
err_alt_u_48 = np.abs(adjusted_bin_option_price_48 - Black_Scholes_Call_Option_Price)
err_tilted_tree_48 = np.abs(tilted_tree_price_48 - Black_Scholes_Call_Option_Price)
err_richardson_price_48 = np.abs(richardson_price_48 - Black_Scholes_Call_Option_Price)

mean_richardson_48 = (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[1]
                       - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M2 / 2), True, False, False, True)[1])
second_moment_richardson_48 = (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[2] 
                       - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M2 / 2), True, False, False, True)[2])

comp_time_CRR_approx_48 = []
for i in range(iterations):
    start_time_CRR_approx_48 = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, False)[0]
    end_time_CRR_approx_48 = time.perf_counter()
    CRR_approx_times_48 = end_time_CRR_approx_48 - start_time_CRR_approx_48
    comp_time_CRR_approx_48.append(CRR_approx_times_48)
comp_time_CRR_approx_48 = sum(comp_time_CRR_approx_48) / iterations

comp_time_CRR_exact_48 = []
for i in range(iterations):
    start_time_CRR_exact_48 = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, True, False)[0]
    end_time_CRR_exact_48 = time.perf_counter()
    CRR_exact_times_48 = end_time_CRR_exact_48 - start_time_CRR_exact_48
    comp_time_CRR_exact_48.append(CRR_exact_times_48)
comp_time_CRR_exact_48 = sum(comp_time_CRR_exact_48) / iterations

comp_time_CRR_alt_48 = []
for i in range(iterations):
    start_time_CRR_alt_48 = time.perf_counter()
    alt_u_CRR(285, 0.021, 5/3, 290, M2, True)[0]
    end_time_CRR_alt_48 = time.perf_counter()
    alt_CRR_times_48 = end_time_CRR_alt_48 - start_time_CRR_alt_48
    comp_time_CRR_alt_48.append(alt_CRR_times_48)
comp_time_CRR_alt_48 = sum(comp_time_CRR_alt_48) / iterations

comp_time_tilted_tree_48 = []
for i in range(iterations):
    start_time_tilted_48 = time.perf_counter()
    binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[0]
    end_time_tilted_48 = time.perf_counter()
    tilted_times_48 = end_time_tilted_48 - start_time_tilted_48
    comp_time_tilted_tree_48.append(tilted_times_48)
comp_time_tilted_tree_48 = sum(comp_time_tilted_tree_48) / iterations

comp_time_richardson_48 = []
for i in range(iterations):
    start_time_richardson_48 = time.perf_counter()
    (2 * binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[0]
                        - binomial_CRR(285, 0.021, 0.12, 5/3, 290, int(M2 / 2), True, False, False, True)[0])
    end_time_richardson_48 = time.perf_counter()
    richardson_times_48 = end_time_richardson_48 - start_time_richardson_48
    comp_time_richardson_48.append(richardson_times_48)
comp_time_richardson_48 = sum(comp_time_richardson_48) / iterations


data_48 = {
    "Price of Call Option": [Black_Scholes_Call_Option_Price,
                             binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, False)[0],
                             exact_crr_price_48, adjusted_bin_option_price_48, tilted_tree_price_48,
                             richardson_price_48],

    "Absolute option Price Error": [0, err_crr_approx_48, err_exact_crr_48, err_alt_u_48, err_tilted_tree_48,
                                    err_richardson_price_48],

    "Mean Stock Price": [Black_Scholes_Call_Mean,
                         binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, False)[1],
                         binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, True, False)[1],
                         adjusted_mean_stock_price_bin_48,
                         binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[1],
                         mean_richardson_48],

    "Second Moment":  [Black_Scholes_Call_Second_Moment,
                       binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, False)[2],
                       binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, True, False)[2],
                       adjusted_second_moment_bin_48,
                       binomial_CRR(285, 0.021, 0.12, 5/3, 290, M2, True, False, False, True)[2],
                       second_moment_richardson_48],
                       

    "Computational Time": [comp_time_BS, comp_time_CRR_approx_48, comp_time_CRR_exact_48, comp_time_CRR_alt_48, comp_time_tilted_tree_48, comp_time_richardson_48]
}

rows_48 = ["Black Scholes (M = 48)", "CRR Approximation (M = 48)", "Exact CRR (M = 48)", "Alternative u Choice (M = 48)", "Tilted Tree (M = 48)", "Richardson Extrapolation (M = 48)"]

df_48 = pd.DataFrame(data_48, index = rows_48)
df_48 = df_48.map('{:.10f}'.format)
print(df_48.to_string())

data_50 = {
    "Price of Call Option": [Black_Scholes_Call_Option_Price,
                             binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, False)[0],
                             exact_crr_price_50, adjusted_bin_option_price_50, tilted_tree_price_50,
                             richardson_price_50],

    "Absolute option Price Error": [0, err_crr_approx_50, err_exact_crr_50, err_alt_u_50, err_tilted_tree_50,
                                    err_richardson_price_50],

    "Mean Stock Price": [Black_Scholes_Call_Mean,
                         binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, False)[1],
                         binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, True, False)[1],
                         adjusted_mean_stock_price_bin_50,
                         binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[1],
                         mean_richardson_50],

    "Second Moment":  [Black_Scholes_Call_Second_Moment,
                       binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, False)[2],
                       binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, True, False)[2],
                       adjusted_second_moment_bin_50,
                       binomial_CRR(285, 0.021, 0.12, 5/3, 290, M1, True, False, False, True)[2],
                       second_moment_richardson_50],
                       

    "Computational Time": [comp_time_BS, comp_time_CRR_approx_50, comp_time_CRR_exact_50, comp_time_CRR_alt_50, comp_time_tilted_tree_50, comp_time_richardson_50]
}

rows_50 = ["Black Scholes (M = 50)", "CRR Approximation (M = 50)", "Exact CRR (M = 50)", "Alternative u Choice (M = 50)", "Tilted Tree (M = 50)", "Richardson Extrapolation (M = 50)"]

df_50 = pd.DataFrame(data_50, index = rows_50)
df_50 = df_50.map('{:.10f}'.format)
print(df_50.to_string())