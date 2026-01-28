import os
import sys

site_packages_path = r"C:\Users\Florian Taube\Downloads\molecule-icon-generator-v2.1.1\lmonari5-molecule-icon-generator-444b51f"
# Ensure the directory exists
if os.path.exists(site_packages_path):
    # Add the site-packages directory to the Python path
    sys.path.append(site_packages_path)
else:
    raise FileNotFoundError(
        f"The specified path does not exist: {site_packages_path}"
    )
import molecule_icon_generator as mig
from rdkit import Chem

# MOLfile einlesen
mol = Chem.MolFromMolFile("Pro_1.mol", removeHs=True)

# Icon erstellen
mig.icon_print(
    mol,
    name="Pro1",
    rdkit_svg=True,
    single_bonds=False,
    rotation=(-20, 50, 90),
)
mol = Chem.MolFromMolFile("Pro_2.mol", removeHs=True)
mig.icon_print(
    mol,
    name="Pro2",
    rdkit_svg=True,
    single_bonds=False,
    rotation=(-20, 40, 60),
)
