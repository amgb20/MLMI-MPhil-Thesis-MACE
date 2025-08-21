from ase.build import bulk, make_supercell
from ase.visualize import view

# 1. Créer une maille primitive de diamant
diamond = bulk('C', 'diamond', a=3.57)  # a = paramètre de maille en Å

# 2. Créer une supercellule (par exemple 3x3x3)
supercell = diamond.repeat((2,2, 2))

# # 3. Visualiser
# # if the visualization is available
# try:
#     view(supercell)
# except Exception as e:
#     pass

# 4. Sauvegarder dans un fichier
from ase.io import write
write("diamond_3x3x3.xyz", supercell)