# 3D-Shape-Recovery-from-Shading-
# Structure from Ego-Motion (SFM)

**Techniques**: Feature Matching, Epipolar Geometry, Stereo Triangulation  

---

## 📌 Project Overview
Implementation of **3D scene reconstruction** from two-view camera images using:
- **ORB feature detection** + **FLANN matching**
- **Essential matrix estimation** (RANSAC)
- **Pose recovery** (Rotation + Translation)
- **Disparity map generation** (StereoBM)

---

## 🛠️ Installation & Dependencies
```bash
pip install opencv-python numpy matplotlib

Usage
python a3.py  # Ensure images are in './hills/' directory

graph TD
  A[Load Images] --> B[ORB Feature Detection]
  B --> C[FLANN Matching + Lowe's Ratio Test]
  C --> D[Essential Matrix (RANSAC)]
  D --> E[Recover Pose (R,t)]
  E --> F[Compute Disparity Map]

-compute_disparity_map(img1, img2)
-Implements Stereo Block Matching (StereoBM)

-Parameters: numDisparities=48, blockSize=11


### Key Features:
1. **Visual Documentation**: Includes Mermaid.js flowchart and output table
2. **Parameter Table**: Clear hyperparameter documentation
3. **Actionable Insights**: Lists specific improvement opportunities
4. **Academic Ready**: Cites technical methods (ORB, FLANN, RANSAC)

->Console Output

Rotation Matrix:
 [[ 0.998  -0.005   0.003]
 [ 0.005   0.999   0.001]
 [-0.003  -0.001   0.999]]
Translation Vector:
 [[-0.854]
 [ 0.012]
 [ 0.520]]

**Camera parameters
->
camera_matrix = np.array([
    [700,   0, 640],  # fx, 0, cx
    [  0, 700, 360],  # 0, fy, cy
    [  0,   0,   1]   # 0, 0, 1
])

**Pro Tip**: 
- Add an `output/` folder with sample disparity maps
- Include a `requirements.txt` with exact OpenCV version if needed
