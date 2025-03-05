import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate synthetic data
np.random.seed(42)
n_points = 100
x = np.linspace(0, 10, n_points)
# True underlying model for probability (using a logistic function)
true_m = 2.0
true_b = -10.0
prob = 1 / (1 + np.exp(-(true_m * x + true_b)))
# Sample binary labels based on the probability
y = (np.random.rand(n_points) < prob).astype(int)

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters for logistic regression
m_param = 0.0  # slope
b_param = 0.0  # intercept
learning_rate = 0.1
n_iter = 500

# Lists to store parameter history for animation
m_history = [m_param]
b_history = [b_param]

# Perform gradient descent for logistic regression
for i in range(n_iter):
    z = m_param * x + b_param
    p = sigmoid(z)
    # Compute gradients (using the derivative of the cross-entropy loss)
    error = p - y
    m_grad = np.mean(error * x)
    b_grad = np.mean(error)
    
    # Update parameters
    m_param -= learning_rate * m_grad
    b_param -= learning_rate * b_grad
    
    # Save parameter history
    m_history.append(m_param)
    b_history.append(b_param)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, c='blue', label='Data', zorder=5)
# Initial logistic curve based on the initial parameters
line, = ax.plot(x, sigmoid(m_history[0] * x + b_history[0]), 'r-', lw=2, label='Logistic Curve')
iter_text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlabel('x')
ax.set_ylabel('Predicted Probability')
ax.set_title('Animated Logistic Regression via Gradient Descent')
ax.legend()
ax.set_ylim(-0.1, 1.1)

# Animation update function
def animate(i):
    current_m = m_history[i]
    current_b = b_history[i]
    y_curve = sigmoid(current_m * x + current_b)
    line.set_ydata(y_curve)
    iter_text.set_text(f"Iteration {i}\nm = {current_m:.2f}, b = {current_b:.2f}")
    return line, iter_text

# Create the animation (300 iterations; update every 50ms)
ani = animation.FuncAnimation(fig, animate, frames=len(m_history), interval=50, blit=True)

plt.show()
