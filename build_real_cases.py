import os
import json
import gzip
import base64
import numpy as np
import nibabel as nib
import scipy.ndimage

# === CONFIGURATION ===
CT_FILE = "AXIAL_1X1_NCE_334394.26545.nii.gz"
MASK_FILE = "segmentation_ct14.nii"
CASE_NAME = "Cas normal 1"              
OUTPUT_FILE = "case_1.json"             
RESIZE_FACTOR = 0.5 
# =====================

# DICTIONNAIRE ÉPURÉ POUR LES EDN (Sans les côtes, muscles superficiels, etc.)
TS_MAP = {
    1: "Rate", 2: "Rein droit", 3: "Rein gauche", 4: "Vésicule biliaire",
    5: "Foie", 6: "Estomac", 7: "Aorte", 8: "Veine cave inférieure",
    9: "Veine porte et splénique", 10: "Pancréas", 11: "Surrénale droite",
    12: "Surrénale gauche", 13: "Lobe supérieur du poumon gauche", 14: "Lobe inférieur du poumon gauche",
    15: "Lobe supérieur du poumon droit", 16: "Lobe moyen du poumon droit", 17: "Lobe inférieur du poumon droit",
    18: "Vertèbre L5", 19: "Vertèbre L4", 20: "Vertèbre L3", 21: "Vertèbre L2",
    22: "Vertèbre L1", 23: "Vertèbre T12", 24: "Vertèbre T11", 25: "Vertèbre T10",
    26: "Vertèbre T9", 27: "Vertèbre T8", 28: "Vertèbre T7", 29: "Vertèbre T6",
    30: "Vertèbre T5", 31: "Vertèbre T4", 32: "Vertèbre T3", 33: "Vertèbre T2",
    34: "Vertèbre T1", 35: "Vertèbre C7", 36: "Vertèbre C6", 37: "Vertèbre C5",
    38: "Vertèbre C4", 39: "Vertèbre C3", 40: "Vertèbre C2", 41: "Vertèbre C1",
    42: "Œsophage", 43: "Trachée", 44: "Myocarde", 45: "Atrium gauche",
    46: "Ventricule gauche", 47: "Atrium droit", 48: "Ventricule droit",
    49: "Artère pulmonaire", 50: "Cerveau", 51: "Artère iliaque gauche", 52: "Artère iliaque droite",
    53: "Veine iliaque gauche", 54: "Veine iliaque droite", 55: "Intestin grêle", 56: "Duodénum",
    57: "Côlon", 
    # 58 à 81 retirés (Côtes)
    82: "Humérus gauche", 83: "Humérus droit", 84: "Scapula gauche", 85: "Scapula droite",
    86: "Clavicule gauche", 87: "Clavicule droite", 88: "Fémur gauche", 89: "Fémur droit",
    90: "Hanche gauche", 91: "Hanche droite", 92: "Sacrum", 
    # 93 à 101 retirés (Face, muscles fessiers et érecteurs)
    102: "Muscle iliopsoas gauche", 103: "Muscle iliopsoas droit",
    104: "Vessie", 105: "Prostate", 106: "Thyroïde", 107: "Moelle spinale",
    108: "Sternum"
    # 109 à 117 retirés (Cartilages, implants, inconnus)
}

def process_case():
    print(f"📦 Traitement du cas : {CASE_NAME}")
    os.makedirs("cases", exist_ok=True)
    
    print("Lecture du scanner...")
    ct_img = nib.load(CT_FILE)
    ct_data = ct_img.get_fdata().astype(np.int16)
    spacing = [float(x) for x in ct_img.header.get_zooms()[:3]]
    
    print(f"Lecture du masque de segmentation ({MASK_FILE})...")
    m_img = nib.load(MASK_FILE)
    label_data = m_img.get_fdata().astype(np.uint8)
    
    structures = []
    unique_labels = np.unique(label_data)
    
    print("Extraction des structures...")
    for val in unique_labels:
        val = int(val)
        
        # Si c'est le fond (0) ou un organe qu'on a retiré de la liste (ex: côtes), on l'ignore
        if val == 0 or val not in TS_MAP:
            continue
        
        struct_name = TS_MAP[val]
        structures.append([val, struct_name])
        print(f"  + Ajout de : {struct_name} (ID: {val})")

    if RESIZE_FACTOR != 1.0:
        print(f"Réduction de la taille (Facteur {RESIZE_FACTOR})...")
        ct_data = scipy.ndimage.zoom(ct_data, (RESIZE_FACTOR, RESIZE_FACTOR, RESIZE_FACTOR), order=1).astype(np.int16)
        label_data = scipy.ndimage.zoom(label_data, (RESIZE_FACTOR, RESIZE_FACTOR, RESIZE_FACTOR), order=0).astype(np.uint8)
        spacing = [s / RESIZE_FACTOR for s in spacing]

    dims = list(ct_data.shape)
    
    print("Compression des données (cela peut prendre 1 à 2 minutes)...")
    ct_gz = gzip.compress(ct_data.flatten('F').tobytes())
    label_gz = gzip.compress(label_data.flatten('F').tobytes())
    
    ct_b64 = base64.b64encode(ct_gz).decode('ascii')
    label_b64 = base64.b64encode(label_gz).decode('ascii')
    
    output_path = os.path.join("cases", OUTPUT_FILE)
    case_data = {
        "n": CASE_NAME,
        "d": dims,
        "sp": spacing,
        "s": structures,
        "ct": ct_b64,
        "b": label_b64
    }
    
    with open(output_path, 'w') as f:
        json.dump(case_data, f)
        
    print(f"✅ Fichier du cas généré : {output_path} ({len(ct_b64)/1024/1024:.1f} Mo)")
    
    index_path = os.path.join("cases", "index.json")
    index_data = []
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            try: index_data = json.load(f)
            except: pass
            
    case_entry = {"n": CASE_NAME, "u": f"cases/{OUTPUT_FILE}"}
    existing = [i for i, c in enumerate(index_data) if c.get("u") == case_entry["u"]]
    if existing: index_data[existing[0]] = case_entry
    else: index_data.append(case_entry)
        
    with open(index_path, 'w') as f:
        json.dump(index_data, f)
        
    print(f"✅ Fichier d'index mis à jour : {index_path}")

if __name__ == "__main__":
    process_case()