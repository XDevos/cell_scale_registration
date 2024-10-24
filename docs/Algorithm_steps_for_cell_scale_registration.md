# Algorithm steps for cell-scale registration

Hypothesis: we can approximate the deformation in the exp49 of David like a linear scale change (zoom).

We need the DAPI masks, DAPI fiducial, other fiducials.

1. Segment DAPI masks
2. Find the center of each mask
3. Extract in the DAPI fiducial a sphere based on this center with the radius of the mask +10~50% depending on the reasonable global shift
4. Extract the same sphere in each other fiducials
5. Cross-correlate other with DAPI cell-scale fiducial
6. Hypothesis: the center of each nucleus are registered
7. convert each cell-scale fiducial in polar domain
8. Compute a cross-correlation registration in polar domain and keep the radius shift
9. Apply the radius shift in the cartesian domain like a rescaling 
10. (assessment) evaluate the registration with a MSE or SSIM
11. generate a vector field to apply a shift for each pixel inside the corresponding DAPI mask without the +10~50%

