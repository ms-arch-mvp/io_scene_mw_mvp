import bpy
import pathlib

# Define the folder path containing .nif files
nif_folder = r"C:/Users/<Username>/AppData/Local/ModOrganizer/Morrowind/overwrite/Export Cells"

# Get all .nif files in the folder
nif_files = pathlib.Path(nif_folder).glob("*.nif")

# Import each .nif file
for nif_file in nif_files:
    bpy.ops.import_scene.mw(
        filepath=str(nif_file),
        use_existing_materials = True,
        ignore_animations = True,        
        ignore_armatures = True,
        ignore_billboard_nodes = True,
        ignore_particle_nodes = False,      
        ignore_emissive_color = False,  
        ignore_tri_shadow = False,
        ignore_nodes = "",
        ignore_nodes_under_switches = "OFF, HARVESTED, Closed",
        filter_best_lod = True,
        use_texture_fallbacks = True,
        use_texture_path_in_material_name = False,
        always_use_file_name_for_root_name = False,
        proxy_mode = False
    )