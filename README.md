<h1 align="center">Morrowind Blender Plugin MVP Scripts</h1>

[Morrowind Visualisation Project](https://ms-arch.gitbook.io/morrowind-visualisation-project) and [Export Cells](https://github.com/ms-arch-mvp/Export_Cells) rely on modifications to the [Morrowind Blender Plugin](https://github.com/Greatness7/io_scene_mw) import scripts. These modifications are designed to be non destructive additions, and also take advantage of Blender 5's longer data-block names. Several additional nodes are included, and the default settings of the importer are to include all handled nodes. Ignore settings can be set to filter the nodes. There are also several safety fixes that aim to prevent import from ever crashes.

To install, extract into: `AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\io_scene_mw`

* `scripts\import_simple.py` can be used for a quick start import, opened in the Scripting tab of Blender.<br>
* `nif_folder` can be pointed to the output of Export Cells or any other folder containing NIF files.
* The import will usually require further merging and hierarchy simplification to make it more usable.

### Requirements

* [Blender 5](https://www.blender.org/)
* [Morrowind Blender Plugin](https://github.com/Greatness7/io_scene_mw)

# Features

### Folder Normalization

* The exporter uses paths to NIFs and these need to be normalized so they are consistent. The references in Morrowind do not have consistency in terms of case or even forward slashes vs backslashes. Normalization assumes forward slashes and lower case.
* It also takes prefixes of up to 3 characters and makes them lower case, otherwise this is inconsistent.
* The rest cannot be normalized without losing the casing, which would make names less readable.

### Name Sanitization

* Added name sanitization to remove unicode characters. This is enables safe imports as otherwise errors may be caused by non unicode characters in NIF files.

### Root Name

* Uses the file name when no root name is present (such as when using exporters from Morrowind) or if `always\_use\_file\_name\_for\_root\_name = True`

### Spatial Object Filtering

* This ensures that only objects with spatial transformations (NiAVObject) are processed as scene nodes, preventing the AttributeError when encountering property blocks at the root or within child lists.

### Lights

* Added light imports.
* Imports lights with the diffuse color and intensity.

### Ignore\_Animations

* Added a check for `ignore\_animations` at the start of `create\_vertex\_morphs()`

### Ignore\_Armatures

* Added ignore\_armatures setting. This is an option to not import armatures for larger scenes.

### Ignore\_Billboards

* Added ignore\_billboards as a new setting.
* NiBillboardNodes can now included on import, as these can contain geometry and effects.

### Ignore\_Emissive\_Color

* Added ignore\_emissive\_color setting. This prevents the emissive color from being imported and can be used to normalize the appearance of meshes, as the Blender Morrowind Plugin does not handle the emissive color correctly in the renderer.

### Ignore\_Shadow\_Meshes

* Added ignore\_shadow\_meshes setting. This can filter out "shadow" meshes typically found under armatures e.g. for creatures.

### Ignore\_Nodes

* Ignore any node and its subtree by name
* Example: `ignore_nodes = "Lightning"`

### Ignore\_Nodes\_Under\_Switches

* Ignores children of a NiSwitchNode by name.
* Example: `ignore_nodes_under_switches = "OFF, HARVESTED, Closed"`
* Should be considered for Glow in the Dahrk, Graphic Herbalism etc.

### Filter\_Best\_LOD

* Only imports the first LOD branch. 

### Empty Root Safeguard

* Added safe guard when root output is missing.
* This can happen if the NIF imports nothing, e.g. the NIF only contains a billboard node that is ignored. Instead of erroring, it is handled.

### Create\_UV\_Controller Safeguard

* Adds validation/coercion around UV key arrays to prevent crashes.

### NiStencilProperty Safeguard

* Implemented a safe\_enum helper function to handle invalid enum values, prevents crashes.

### No Texture Material Names

* Added handling of material name with no textures when used with use\_existing\_materials
* Previously, this would just fallback to generic material naming which does not help distinguish materials. Now, it falls back to the material property.

### Material Names Inclusions

* Various inclusions and exclusions to make material name more specific.
* Include `diffuse:` if non default. Default is `#ffffff`
  \
  or
  \
  Include `diffuse:Col` if the material is set to use Vertex Colors
* Include `emissive:` if non default. Default is `#000000`
* Include `alpha:` if non default. Combines NiAlphaProperty blending mode and NiMaterialProperty (Opacity)
* Exclude `decal\_1`, `decal\_2`, etc. These make the material name too long and are generally not very useful. These may be used for textures such as water caustics.
* The goal is to maintain a level of uniqueness when deduplicating materials, while still grouping materials where appropriate.

### Use\_Texture\_Fallbacks

* Mimics the standard behaviour of Morrowind: if it doesn't find DDS textures, then it looks for TGA, then BMP.
 
### Use\_Texture\_Path\_In\_Material\_Name

* Include path: if non default. Adds the path to the folder of the base texture.
* This option has been provided and is false by default. It prevents deduplication when the texture names but paths are different. In many cases, deduplicating these is actually useful because these textures are often identical and are duplicated. It can also help unify materials to one source of truth for the textures.

### Proxy Mode

* Imports the file as cubes for the first mesh in every nif for fast imports, debugs and processing.
