from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


def simulate_pid(Kp, Ki, Kd):
    dt = 0.01
    t_final = 20
    t = np.arange(0, t_final, dt)

    a = 10.0
    b = 0.3
    P_set = -16.7  # -125 mmHg ≈ -16.7 kPa

    P = np.zeros_like(t)
    u = np.zeros_like(t)

    integral = 0
    previous_error = 0

    for i in range(1, len(t)):
        error = P_set - P[i - 1]
        integral += error * dt
        derivative = (error - previous_error) / dt

        control = Kp * error + Ki * integral + Kd * derivative

        # Vacuum pump action is inverted
        u[i] = max(0, min(1, -control))

        dPdt = -a * u[i] - b * P[i - 1]
        P[i] = P[i - 1] + dPdt * dt

        previous_error = error

    fig, ax = plt.subplots()
    ax.plot(t, P, label="Pressure response")
    ax.axhline(P_set, linestyle="--", label="Setpoint (-125 mmHg)")
    ax.set_title("PID Vacuum Pressure Response")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [kPa]")
    ax.grid(True)
    ax.legend()

    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
    img.seek(0)
    plt.close(fig)

    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    return plot_url


@app.route("/", methods=["GET", "POST"])
def index():
    Kp = 0.3
    Ki = 0.1
    Kd = 0.02

    if request.method == "POST":
        Kp = float(request.form["Kp"])
        Ki = float(request.form["Ki"])
        Kd = float(request.form["Kd"])

    plot_url = simulate_pid(Kp, Ki, Kd)

    return render_template(
        "index.html",
        plot_url=plot_url,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd
    )


if __name__ == "__main__":
    app.run(debug=True)
