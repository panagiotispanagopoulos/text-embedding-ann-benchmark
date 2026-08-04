import matplotlib.pyplot as plt

# -----------------------------
# DATA από τα experiments σου
# -----------------------------

# nprobe tuning
nprobe = [1, 3, 5, 10, 20]
recall = [0.76, 0.88, 0.90, 0.96, 0.98]
latency_nprobe = [0.004556, 0.004470, 0.004743, 0.004523, 0.004527]

# scaling experiment
sizes = [1000, 2000, 5000]
exact_latency = [0.007801, 0.007745, 0.011218]
faiss_latency = [0.005670, 0.005835, 0.005755]
speedup = [1.3757, 1.3275, 1.9491]
recall_scaling = [1.00, 0.98, 0.96]

# -----------------------------
# Plot 1: Recall vs nprobe
# -----------------------------
plt.figure()
plt.plot(nprobe, recall, marker='o')
plt.xlabel("nprobe")
plt.ylabel("Recall@5")
plt.title("Recall vs nprobe")
plt.grid()
plt.show()

# -----------------------------
# Plot 2: Latency vs Dataset Size
# -----------------------------
plt.figure()
plt.plot(sizes, exact_latency, marker='o', label="Exact")
plt.plot(sizes, faiss_latency, marker='o', label="FAISS")
plt.xlabel("Dataset Size")
plt.ylabel("Latency (seconds)")
plt.title("Latency vs Dataset Size")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Plot 3: Speedup vs Dataset Size
# -----------------------------
plt.figure()
plt.plot(sizes, speedup, marker='o')
plt.xlabel("Dataset Size")
plt.ylabel("Speedup (Exact / FAISS)")
plt.title("Speedup vs Dataset Size")
plt.grid()
plt.show()

# -----------------------------
# Plot 4: Recall vs Dataset Size
# -----------------------------
plt.figure()
plt.plot(sizes, recall_scaling, marker='o')
plt.xlabel("Dataset Size")
plt.ylabel("Recall@5")
plt.title("Recall vs Dataset Size")
plt.grid()
plt.show()