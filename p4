# Gradient Descent for y = (x + 3)^2, starting at x = 2

def f(x):
    return (x + 3) ** 2          # the function

def gradient(x):
    return 2 * (x + 3)           # its derivative dy/dx

x = 2.0                          # starting point
learning_rate = 0.1
iterations = 50

print("Iter      x          y        gradient")
for i in range(iterations + 1):
    if i < 6 or i % 10 == 0:
        print("%3d   %9.5f  %9.5f  %9.5f" % (i, x, f(x), gradient(x)))
    x = x - learning_rate * gradient(x)      # the update rule

print("\nLocal minimum found at x = %.5f, y = %.8f" % (x, f(x)))
