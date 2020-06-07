from scipy.stats import skewnorm
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,2.5))
x1 = np.linspace(skewnorm.ppf(0.01, -3), skewnorm.ppf(0.99, -3),100)
x2 = np.linspace(skewnorm.ppf(0.01, 0), skewnorm.ppf(0.99, 0),100)
x3 = np.linspace(skewnorm.ppf(0.01, 3), skewnorm.ppf(0.99, 3),100)
ax1.plot(skewnorm(-3).pdf(x1),'k-',lw=4)
ax2.plot(skewnorm(0).pdf(x2),'k-',lw=4)
ax3.plot(skewnorm(3).pdf(x3),'k-',lw=4)


#kurt

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,2.5))
axs = [ax1, ax2, ax3]
Titles = ["Mesokurtic", "Lepkurtic", "Playkurtic"]

#Normal Distoribution
dist = scipy.stats.norm(loc=100, scale=5)
sample_norm = dist.rvs(size=10000)
# Leptokurtic Distoribution
dist2 = scipy.stats.laplace(loc=100, scale=5)
sample_laplace = dist2.rvs(size=10000)

dist3 = scipy.stats.cosine(loc=100, scale=5)
sample_cosine = dist3.rvs(size=10000)

samples = [sample_norm, sample_laplace, sample_cosine]

for n in range(0, len(axs)):
    axs[n].hist(samples[n], bins='auto', normed=True)
    axs[n].set_title('{}'.format(Titles[n]))
    print("Kurtosis of" + Titles[n])
    print(scipy.stats.describe(samples[n])[5])
