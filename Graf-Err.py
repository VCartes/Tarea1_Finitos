import matplotlib.pyplot as plt

deg = 3

L2_H1 = 2

hh = list()
err = list()

for eps in [0.1, 0.01, 0.001]:
    file_name = f"Tables/deg {deg} - eps {eps}.txt"
    file = open(file_name, "r")

    hh = list()
    err = list()

    for line in file:
        items = line.strip().split(" ")
        hh.append(float(items[0]))
        err.append(float(items[L2_H1]))

    plt.loglog(hh, err, label=f"ε = {eps}")

ylab = ["L2", "H1"]
plt.xlabel("h")
plt.ylabel(f"error {ylab[L2_H1-1]}")
plt.legend()
plt.title(f"Error {ylab[L2_H1-1]} para k = {deg}")

plt.savefig(f"Graf Err/deg {deg} - {ylab[L2_H1-1]}.png")

plt.show()
