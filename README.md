<h1 align="center">Morrowind Blender Plugin MVP Scripts</h1>

[Morrowind Visualisation Project](https://ms-arch.gitbook.io/morrowind-visualisation-project) and [Export Cells](https://github.com/ms-arch-mvp/Export_Cells) rely on modifications to the [Morrowind Blender Plugin](https://github.com/Greatness7/io_scene_mw) import scripts. These modifications are designed to be non destructive additions, and also take advantage of Blender 5's longer data-block names.

To install, extract into: `AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\io_scene_mw`

* `scripts\import_simple.py` can be used for a quick start import, opened in the Scripting tab of Blender.<br>
* `nif_folder` can be pointed to the output of Export Cells or any other folder containing NIF files.
* The import will usually require further merging and hierarchy simplification to make it more usable.

### Requirements

* [Blender 5](https://www.blender.org/)
* [Morrowind Blender Plugin](https://github.com/Greatness7/io_scene_mw)

# Features

### Folder Normalization

* The exporter uses paths to NIFs and these need to be normalized so they are consistent. The references in Morrowind do not have consistency in terms of case or even forward slashes vs backslashes. Noramlization assumes forward slashes and lower case.
* It also takes prefixes of up to 3 characters and makes them lower case, otherwise this is inconsistent.
* The rest cannot be normalized without losing the casing, which would make names less readable.

### Name Sanitization

* Added name sanitization to remove unicode characters. This is enables safe imports as otherwise errors may be caused by non unicode characters in .nif files.

### Spatial Object Filtering

* This ensures that only objects with spatial transformations (NiAVObject) are processed as scene nodes, preventing the AttributeError when encountering property blocks at the root or within child lists.

### Lights

* Added light imports. This imports lights with the diffuse color and intensity. Lights can be exported using glTF and enablign the include Punctual Lights option.

### Ignore\_Animations Modification

* Added a check for ignore\_animations at the start of create\_vertex\_morphs()
* Ignoring animations is required, otherwise imports will fail. This extra check ensures no animations at all are imported.

### NiStencilProperty Modification

* Implemented a safe\_enum helper function to handle invalid enum values, prevents crashes.

### Ignore\_Armatures

* Added ignore\_armatures setting. This is an option to not import armatures for larger scenes.
* Added to nif\_import.py and operators/import\_scene.py
* This is required as multiple armatures are not yet supported by the plugin.

### Ignore\_Billboards

* Added ignore\_billboards as a new setting.
* NiBillboardNodes can now included on import, as these can contain geometry and effects.

### Ignore\_Shadow\_Meshes

* Added ignore\_shadow\_meshes setting. This can filter out "shadow" meshes typically found under armatures e.g. for creatures.

### Ignore\_Switch\_Names

* Ignores nodes under switches with given names to simplify NIFs
* Example: ignore_switch_names = "OFF, HARVESTED, Closed"
* Should be considered for Glow in the Dahrk, Graphic Herbalism etc.

### Filter\_Best\_LOD

* Only imports the first LOD branch. 

### Root Name

* Added always\_use\_file\_name\_for\_root\_name.
* Added to nif\_import.py and operators/import\_scene.py
* Uses the file name when no root name is present (such as when using exporters from Morrowind) or if always\_use\_file\_name\_for\_root\_name = True

### No Texture Material Names

* Added handling of material name with no textures when used with use\_existing\_materials
* Previously, this would just fallback to generic material naming which does not help distinguish materials. Now, it falls back to the material property.

### Material Names Inclusions

* Various inclusions and exclusions to make material name more specific.
* Include Diffuse: if non default. Default is #ffffff
  \
  or
  \
  Include diffuse:Col if the material is set to use Vertex Colors
* Include emissive: if non default. Default is #000000
* Include alpha: if non default. Combines NiAlphaProperty blending mode and NiMaterialProperty (Opacity)
* Exclude Decal\_1, Decal\_2, etc. These make the material name too long and are generally not very useful. These may be used for textures such as water caustics.
* The goal is to maintain a level of uniqueness when deduplicating materials, while still grouping materials where appropriate.

**use\_texture\_path\_in\_material\_name**

* Include path: if non default. Adds the path to the folder of the base texture.
* This option has been provided and is false by default. It prevents deduplication when the texture names but paths are different. In many cases, deduplicating these is actually useful because these textures are often identical and are duplicated. It can also help unify materials to one source of truth for the textures.

### Proxy Mode

* Imports the file as cubes for the first mesh in every nif for fast imports, debugs and processing.
