import numpy as np

def make_bins(zvals, min_per_bin=50, n_start=100):
    edges = np.linspace(zvals.min(), zvals.max(), n_start + 1)
    counts, _ = np.histogram(zvals, edges)

    edges = list(edges)
    counts = list(counts)

    i = 0
    while i < len(counts):
        if counts[i] < min_per_bin:
            if i == 0:
                counts[i + 1] += counts[i]
                del counts[i]
                del edges[i + 1]
            else:
                counts[i - 1] += counts[i]
                del counts[i]
                del edges[i]
                i -= 1
        else:
            i += 1

    return np.array(edges)


def find_SNR(image):
    return np.sqrt(np.max(image))
