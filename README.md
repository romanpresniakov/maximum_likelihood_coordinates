This project aims to implement the Maximum Likelihood Coordinates approach presented by Chang et al. (2023)

# 2D and 3D Deformation Viewer

Compute 2D or 3D maximum likelihood coordinates from a cage and apply the deformation to a model.

## Before starting

1. Install dependencies based on requirements defined in the development folder using
```bash
pip install -r requirements.txt
```

## Usage 2D
```bash
python -m development.viz.viewer --shape snake --method smlc 
```

Also, a star shape is available. In order to use normal MLC coordinates use 
```bash
--method mlc.
```

## Usage 3D
```bash
python -m development.viz.viewer3d development/viz/data/horse/horse.obj development/viz/data/horse/bounding_meshb.obj development/viz/data/horse/deformed_bounding_mesh1b.obj
```

You could also use your own (not predefined) OBJ-files. In order to do so, specify the path to the files when running the script. The first argument is a model, the second argument is the cage and the third one is a deformation.  
