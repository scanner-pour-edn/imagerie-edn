import os
import json
import gzip
import base64
import numpy as np
import nibabel as nib
import scipy.ndimage


# === CONFIGURATION ===
# Mets ici le nom de ton scanner et de ton fichier de masque multi-labels
CT_FILE = "AXIAL_1X1_NCE_334394.26545.nii.gz"          # Remplace par le nom de ton fichier CT
MASK_FILE = "segmentation_ct14.nii"     # Ton fichier contenant toutes les segmentations
CASE_NAME = "Cas normal 1"              # Le nom qui s'affichera sur le site
OUTPUT_FILE = "case_1.json"             # Le nom du fichier généré


# DOWNSAMPLE : Réduire la résolution pour le web (0.5 = taille divisée par 2, poids divisé par 8)
RESIZE_FACTOR = 0.5 
# =====================


# Dictionnaire des ID principaux de TotalSegmentator (pour que le site les reconnaisse)
TS_MAP = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate", 
    50: "heart", 51: "aorta", 52: "inferior_vena_cava", 53: "portal_vein_and_splenic_vein",
    58: "brain", 59: "skull", 92: "femur_left", 93: "femur_right",
    104: "autochthon_left", 105: "autochthon_right", 106: "iliopsoas_left", 107: "iliopsoas_right"
}


def process_case():
    print(f"📦 Traitement du cas : {CASE_NAME}")
    os.makedirs("cases", exist_ok=True)
    
    # 1. Charger le CT
    print("Lecture du scanner...")
    ct_img = nib.load(CT_FILE)
    ct_data = ct_img.get_fdata().astype(np.int16)
    spacing = [float(x) for x in ct_img.header.get_zooms()[:3]]
    
    # 2. Charger le masque
    print(f"Lecture du masque de segmentation ({MASK_FILE})...")
    m_img = nib.load(MASK_FILE)
    label_data = m_img.get_fdata().astype(np.uint8)
    
    structures = []
    unique_labels = np.unique(label_data)
    
    # 3. Extraire les structures
    print("Extraction des structures...")
    for val in unique_labels:
        val = int(val)
        if val == 0:
            continue
        # Récupère le nom en anglais pour faire le lien avec le code JS
        struct_name = TS_MAP.get(val, f"organe_{val}")
        structures.append([val, struct_name])
        print(f"  + Ajout de : {struct_name} (ID: {val})")


    # 4. Redimensionnement (Downsampling)
    if RESIZE_FACTOR != 1.0:
        print(f"Réduction de la taille (Facteur {RESIZE_FACTOR})...")
        ct_data = scipy.ndimage.zoom(ct_data, (RESIZE_FACTOR, RESIZE_FACTOR, RESIZE_FACTOR), order=1).astype(np.int16)
        label_data = scipy.ndimage.zoom(label_data, (RESIZE_FACTOR, RESIZE_FACTOR, RESIZE_FACTOR), order=0).astype(np.uint8)
        spacing = [s / RESIZE_FACTOR for s in spacing]


    dims = list(ct_data.shape)
    
    # 5. Compression
    print("Compression des données (cela peut prendre 1 à 2 minutes)...")
    ct_gz = gzip.compress(ct_data.flatten('F').tobytes())
    label_gz = gzip.compress(label_data.flatten('F').tobytes())
    
    ct_b64 = base64.b64encode(ct_gz).decode('ascii')
    label_b64 = base64.b64encode(label_gz).decode('ascii')
    
    # 6. Sauvegarde JSON
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
    
    # 7. Index
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