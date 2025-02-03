import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_curve(metrics, title="Training Curve"):
    fig, ax = plt.subplots()
    ax.plot(metrics, marker='o', linestyle='-', color='blue', label='Score')
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1])
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    buf.close()
    return img_base64
