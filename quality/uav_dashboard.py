"""
uav_dashboard.py
----------------------------------------
Professional UAV Quality Dashboard

AgroFedVision

Author: Amit Mishra
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ==========================================================
# Horizontal Quality Bar Chart
# ==========================================================

def create_quality_dashboard(metrics):

    labels = list(metrics.keys())
    values = list(metrics.values())

    colors = []

    for v in values:

        if v >= 90:
            colors.append("green")

        elif v >= 75:
            colors.append("limegreen")

        elif v >= 60:
            colors.append("orange")

        else:
            colors.append("red")

    plt.figure(figsize=(10,6))

    plt.barh(labels, values, color=colors)

    for i, v in enumerate(values):

        plt.text(v+1, i, f"{v:.1f}%", fontsize=11)

    plt.xlim(0,100)

    plt.xlabel("Quality (%)")

    plt.title("AgroFedVision UAV Quality Assessment")

    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS/"uav_quality_dashboard.png",
        dpi=250
    )

    plt.close()


# ==========================================================
# Nutrient Distribution
# ==========================================================

def create_pie_chart(data):

    labels = list(data.keys())

    values = list(data.values())

    colors = [
        "#4CAF50",
        "#FF9800",
        "#03A9F4",
        "#E91E63"
    ]

    plt.figure(figsize=(6,6))

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )

    plt.title("Nutrient Distribution")

    plt.savefig(
        RESULTS/"uav_nutrient_distribution.png",
        dpi=250
    )

    plt.close()


# ==========================================================
# Reliability Gauge
# ==========================================================

def create_reliability_gauge(score):

    fig, ax = plt.subplots(figsize=(6,3))

    ax.barh(
        [0],
        [100],
        color="lightgray",
        height=0.35
    )

    if score > 90:
        color="green"

    elif score > 75:
        color="limegreen"

    elif score > 60:
        color="orange"

    else:
        color="red"

    ax.barh(
        [0],
        [score],
        color=color,
        height=0.35
    )

    ax.set_xlim(0,100)

    ax.set_yticks([])

    ax.set_xlabel("Reliability (%)")

    plt.title("Overall UAV Reliability")

    plt.text(
        score/2,
        0,
        f"{score:.1f}%",
        ha="center",
        va="center",
        fontsize=18,
        color="white",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS/"uav_reliability_gauge.png",
        dpi=250
    )

    plt.close()


# ==========================================================
# Radar Chart
# ==========================================================

def create_radar(metrics):

    labels = list(metrics.keys())

    values = list(metrics.values())

    values = [v/100 for v in values]

    N = len(labels)

    angles = np.linspace(
        0,
        2*np.pi,
        N,
        endpoint=False
    )

    values += values[:1]
    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=(6,6))

    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values,
        linewidth=2
    )

    ax.fill(
        angles,
        values,
        alpha=0.25
    )

    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(labels)

    plt.title("UAV Quality Radar")

    plt.savefig(
        RESULTS/"uav_quality_radar.png",
        dpi=250
    )

    plt.close()


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    metrics = {

        "Pipeline":100,

        "Healthy":82,

        "Disease":76,

        "Prescription":92,

        "Vegetation":88,

        "Image":90

    }

    nutrients = {

        "Healthy":23.8,

        "Nitrogen":10.5,

        "Phosphorus":14.5,

        "Potassium":51.2

    }

    create_quality_dashboard(metrics)

    create_pie_chart(nutrients)

    create_reliability_gauge(91.4)

    create_radar(metrics)

    print()

    print("Saved")

    print("results/uav_quality_dashboard.png")

    print("results/uav_nutrient_distribution.png")

    print("results/uav_reliability_gauge.png")

    print("results/uav_quality_radar.png")