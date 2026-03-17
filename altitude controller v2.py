import control as ct
import numpy as np
import matplotlib.pyplot as plt

# ---
#Plant fuction governing the physics of the uav's altitude, expressed in laplace trasnformed form.
# ---

m = 1.2 #mass of uav
b = 0.8 #drag coefficient

plant = ct.TransferFunction([1], [m, b, 0])

# ---
#Ziegler–Nichols method for obtaining PID coefficents for transfer function
# ---

Ku = 0
Pu = 0

Kp_values = np.linspace(0.1, 60, 60)

for Kp_test in Kp_values:

    controller = ct.TransferFunction([Kp_test], [1])
    system = ct.feedback(controller * plant)

    t = np.linspace(0, 50, 5000)
    u = np.zeros_like(t)
    u[t>10] = 0.5

    t, y = ct.forced_response(system, t, u )

    
    plt.plot(t, y, label = f"Kp = {Kp_test}")
    plt.legend()
    plt.title(f"k_p value = {Kp_test}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    

    peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0]  # peak detection

    if len(peaks) >= 15:

        periods = np.diff(t[peaks])

        if np.std(periods) < 0.1:   # roughly  constant period
            Ku = Kp_test
            Pu = np.mean(periods)

            Ku_plot = plt.figure()
            plt.plot(t, y, label = f"Ku = {Ku}")
            plt.legend()
            plt.title(f"Ultimate gain = {Ku}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            
            break
        


if Ku is None:
    raise ValueError("Could not determine Ku automatically")

print("Ultimate Gain Ku:", Ku)
print("Oscillation Period Pu:", Pu)


Kp = 0.6 * Ku
Ki = 1.2 * Ku / Pu
Kd = 0.075 * Ku * Pu

print("\nPID Gains")
print("Kp =", Kp)
print("Ki =", Ki)
print("Kd =", Kd)


# Build PID controller


pid = ct.TransferFunction([Kd, Kp, Ki], [1, 0])

closed_loop = ct.feedback(pid * plant)


# Simulate altitude response


t = np.linspace(0, 50, 2000)
u = np.zeros_like(t)
u[t>10] = 1

t, altitude = ct.forced_response(closed_loop, t, u)
print(t)
print(altitude)
error = u - altitude
print("error =", error)

# error plot

error_plot =plt.figure()
plt.plot(t,error)
plt.title("Error_plot")
plt.xlabel("Time (s)")
plt.ylabel("error")
plt.grid(True)


# Plot result

PID = plt.figure()
plt.plot(t, altitude)
plt.title("UAV Altitude Control (Ziegler–Nichols PID)")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid(True)


plt.show()

