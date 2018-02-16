import numpy as np
import matplotlib.pyplot as plt

base = np.array([10.,5.,1.,0.8,0.5,0.3,0.1,0.08,0.05])
cells = np.array([2388,2880,5157,5863,7641,10642,24993,29928,45192])

cd = np.array([0.00786664,0.00525183,0.00344194,0.00334354,0.00322755,0.00312063,0.00293668,0.00291113,0.00287018])
cl = np.array([1.27767,1.28969,1.30093,1.30122,1.30278,1.30463,1.3067,1.30655,1.30695])

# plt.figure(1)
# plt.plot(cells,cd)
# plt.title('drag')

plt.figure(2)
plt.semilogx(1./base,cl,'ob')
# plt.plot(cells,cl,'ob')
plt.title('Grid Convergence',fontsize=20)
plt.xlabel(r'base size$^{-1}$',fontsize=20)
plt.ylabel('Lift Coefficient',fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig('Convergence_Plot.pdf',transparent=True)

plt.show()
