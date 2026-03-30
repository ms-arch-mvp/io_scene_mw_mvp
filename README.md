<h1 align="center">Morrowind Blender Plugin MVP Scripts</h1>

<p align="center">
  <a href="https://ms-arch.gitbook.io/morrowind-visualisation-project/io_scene_mw_mvp/functions">Documentation</a>
</p>

These are modified import scripts for the Morrowind Blender Plugin. Several additional nodes are included, and the default settings of the importer are to include all handled nodes. Ignore settings can be set to filter the nodes. There are also several safeguards that aim to prevent the import from ever crashing. These modifications are designed to be non destructive additions, and also take advantage of Blender 5's longer data-block names.

The [Morrowind Visusalisation Project](https://ms-arch.gitbook.io/morrowind-visualisation-project) and associated tools are reliant on these scripts.

To install, extract into: `AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\io_scene_mw`

* `scripts\import_simple.py` can be used for a quick start import, opened in the Scripting tab of Blender.<br>
* `nif_folder` can be pointed to the output of Export Cells or any other folder containing NIF files.
* The import will usually require further merging and hierarchy simplification to make it more usable.

### Requirements

* [Blender 5](https://www.blender.org/)
* [Morrowind Blender Plugin](https://github.com/Greatness7/io_scene_mw)
