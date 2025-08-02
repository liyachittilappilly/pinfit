# PinFit: Body Shape Detection Project

## 🌟 Overview
PinFit is a Python-based prototype that attempts to classify body shapes (Hourglass, Rectangle, etc.) by analyzing 2D skeletal keypoints from webcam images. Inspired by the vision of a personal fashion assistant, this project uses computer vision to estimate body proportions - with important limitations acknowledged.

> **"Innovation starts somewhere! This is Version 1: a skeletal-based classifier that gives a rough idea of body shape. Not flawless, but fabulous for a prototype."** 💅

## ✨ Features
- Real-time body shape classification via webcam
- Analysis of both front and side views for improved accuracy
- Skeletal keypoint detection using MediaPipe
- Simple visualization of detected body proportions
- Rough shape categorization (Hourglass, Rectangle, etc.)

## ⚠️ Current Limitations
This project has important technical constraints that users should understand:

1. **Not 3D Measurement**: 
   - Does not capture actual 3D body measurements
   - Only calculates ratios between 2D skeletal keypoints (shoulders, waist, hips)

2. **No Soft Tissue Detection**:
   - Cannot detect soft tissues or body volume with standard webcam
   - Classifications based on bone structure and positioning only
   - Labels like "bust" or "waist" are approximations, not medically accurate

3. **Hardware Constraints**:
   - Standard webcams cannot provide depth information
   - Requires specialized hardware (depth cameras, 3D scanners) for true volume analysis

## 🚀 Future Improvements
Planned enhancements to address current limitations:

1. **Depth Sensing Integration**:
   - Implement support for Azure Kinect or Intel RealSense
   - Capture actual 3D measurements for improved accuracy

2. **Advanced AI Models**:
   - Develop neural networks to infer body volume over time
   - Create more sophisticated classification algorithms

3. **Smart Mirror Vision**:
   - Evolve toward a "smart mirror" application for fashion and fitness
   - Real-time style recommendations based on accurate body analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam access
- (Optional) Depth-sensing camera for future versions

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/pinfit.git
cd pinfit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the main application
python main.py

# Follow on-screen instructions to position yourself
# The system will analyze both front and side views
```

## 📊 Technical Details
- **Core Library**: MediaPipe for pose detection
- **Analysis Method**: Ratio calculations between skeletal keypoints
- **Views Processed**: Front and side for improved accuracy
- **Output**: Rough body shape classification

## 🤝 Contributing
We welcome contributions to help improve PinFit! Areas of particular interest:
- Integration with depth-sensing hardware
- Development of volume inference algorithms
- Expansion of training data for diverse body types
- UI/UX improvements

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- MediaPipe team for their excellent pose detection library
- The computer vision community for inspiration and guidance
- Everyone working to make technology more inclusive and personalized

## 📬 Connect
If you're working on similar body-aware applications or have experience with 3D scanning in Python, **I'd love to connect!** Let's make AI understand humans *just a little better*.


---

*"Tech can be glam if you're committed" 💕*  
*Let's build the future of body-aware technology together!*
