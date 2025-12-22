# Predictive Modeling Of L-PBF Processing Maps

### Executive Summary

This project details the development of a predictive model for Laser Powder Bed Fusion (L-PBF)
processing maps, aimed at minimizing the extensive experimental workload required for parameter
optimization. The methodology utilizes the Eagar-Tsai and Gladush-Smurov analytical models to
estimate melt pool dimensions, while simultaneously evaluating defect criteria for balling, keyhole
porosity, and lack of fusion. Implemented in Python, the tool predicts optimal processing windows
to ensure high-quality part fabrication based on specific material properties and laser parameters.

### Introduction

Laser Powder Bed Fusion (L-PBF) is an additive manufacturing technique that utilizes a highenergy
laser beam to selectively melt and fuse metallic powders layer by layer, creating complex
three-dimensional parts directly from digital models. The process involves spreading a thin layer of
metal powder onto the build plate, followed by the laser scanning the powder bed according to the
desired geometry. The laser’s energy melts the powder particles, which then solidify upon cooling
to form a solid layer. This process is repeated layer by layer until the entire part is built.
The development of new materials for L-PBF is often hindered by the extensive experimental
work required to identify optimal processing parameters that yield high-quality parts. To address
this challenge, predictive modeling of L-PBF processing maps has emerged as a valuable tool. These
models aim to establish relationships between processing parameters, such as laser power, scanning
speed, hatch spacing, and layer thickness, and the resulting melt pool characteristics. By accurately
predicting melt pool behavior, these models can help identify optimal processing windows that
minimize defects and significantly reduce the experimental workload that traditionally accompanies
material development for L-PBF.
Zhu et al. [1] developed a predictive analytical model to construct processing maps for Nitinol
alloys in L-PBF. By analyzing the predicted melt pool dimensions and thermal profiles, they were
able to identify regions in the processing parameter space that are likely to yield high-quality parts
with minimal defects.