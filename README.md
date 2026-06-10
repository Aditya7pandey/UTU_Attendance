# FaceMark Pro - Advanced Attendance Management System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**AI-Powered Attendance System with Anti-Spoof Detection**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [API](#api) • [Contributing](#contributing)

</div>

## 🎯 Overview

FaceMark Pro is a comprehensive attendance management system that combines facial recognition, QR code scanning, and blockchain technology to provide secure, accurate, and tamper-proof attendance tracking. Built with advanced anti-spoof detection and real-time analytics.

##  Features

### 🔐 Security & Authentication
- **Anti-Spoof Detection**: Real-time liveness detection using blink analysis
- **Blockchain Integration**: Attendance records are stored on a blockchain (Ganache) to ensure tamper-proof and immutable data storage
- **Face Recognition**: Advanced facial recognition with 95%+ accuracy
- **QR Code Verification**: Dynamic session-based QR codes

###  Analytics & Reporting
- **AI-Powered Insights**: Intelligent attendance pattern analysis
- **Real-time Dashboard**: Live attendance metrics and statistics
- **Automated Reports**: PDF generation with charts and analytics
- **Parent Notifications**: Automated email alerts for low attendance

###  Connectivity
- **Offline Mode**: Works without internet connection
- **Auto-Sync**: Automatic synchronization when online
- **Cloud Storage**: Supabase integration for data persistence
- **Email Integration**: SMTP notifications and alerts

###  User Experience
- **Modern UI**: Gradient-based responsive interface
- **Interactive Chatbot**: AI assistant for attendance queries
- **Multi-Modal Input**: Face, QR code, and manual entry options
- **Real-time Feedback**: Live camera feed with status indicators

##  Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera access
- Internet connection (for cloud features)

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/UTU_Attendance-main.git
cd UTU_Attendance-main
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
# Create .env file with your credentials
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
SENDER_EMAIL=your_gmail@gmail.com
SENDER_PASSWORD=your_app_password
```

4. **Run the application**
```bash
streamlit run main.py
```

## 📋 Usage

### 🎓 Student Registration

1. **Capture Face Image**
   - Click "Register New Face" in sidebar
   - Position face in camera frame
   - Wait for 4-second capture

2. **Fill Registration Form**
   - Student ID and personal details
   - Parent/Guardian information
   - Contact details for notifications

3. **QR Code Generation**
   - Automatic QR code creation
   - Image upload to cloud storage
   - Downloadable QR code for student

### ✅ Attendance Marking

#### Face Recognition Mode
- **Live Detection**: Real-time face recognition
- **Anti-Spoof**: Blink verification required
- **Automatic Marking**: Instant attendance logging

#### QR Code Mode
- **Session-Based**: Teacher starts attendance session
- **Dynamic QR**: Auto-refreshing QR codes every 5 seconds
- **Student Scanning**: Students scan with their devices

#### Manual Entry
- **Backup Option**: Manual attendance entry
- **Verification**: Cross-check with registered students

### 📊 Analytics Dashboard

#### Real-time Metrics
- Total class days
- Today's attendance count
- Registered students
- Gold badge holders

#### Reports & Insights
- **AI Analysis**: Intelligent attendance patterns
- **Attendance Percentage**: Individual student tracking
- **Low Attendance Alerts**: Automated parent notifications
- **PDF Reports**: Comprehensive attendance reports

## 🏗️ Architecture

### Core Components

```
FaceMark Pro/
├── main.py                 # Main Streamlit application
├── supabase_client.py      # Database connection
├── blockchain_attendance.py # Blockchain integration
├── student_app.py          # Student-specific features
├── offline_sync.py         # Offline functionality
├── Training_Images/        # Face recognition dataset
├── QR_Codes/              # Generated QR codes
└── requirements.txt       # Dependencies
```

### Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Computer Vision**: OpenCV, face_recognition, MediaPipe
- **Database**: Supabase (PostgreSQL)
- **Blockchain**: Custom implementation
- **AI/ML**: Groq API for insights
- **Email**: SMTP with HTML templates
- **QR Codes**: qrcode, pyzbar libraries

## 🔧 Configuration

### Database Setup (Supabase)

Create the following tables in your Supabase project:

```sql
-- Students table
CREATE TABLE students_data (
    id SERIAL PRIMARY KEY,
    Student_ID VARCHAR(50) UNIQUE,
    Name VARCHAR(100),
    Age INTEGER,
    Grade VARCHAR(50),
    Parent_Name VARCHAR(100),
    Parent_Contact VARCHAR(20),
    Parent_Gmail VARCHAR(100)
);

-- Attendance table
CREATE TABLE Attendance (
    id SERIAL PRIMARY KEY,
    Name VARCHAR(100),
    Date DATE,
    Time TIME,
    Method VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rewards table
CREATE TABLE rewards (
    id SERIAL PRIMARY KEY,
    Name VARCHAR(100),
    AttendanceCount INTEGER DEFAULT 0,
    Badge VARCHAR(20) DEFAULT 'No Badge'
);
```

### Email Configuration

1. **Enable 2FA** on your Gmail account
2. **Generate App Password** in Google Account settings
3. **Update credentials** in the configuration section












## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **Streamlit** for the web framework
- **Supabase** for backend services
- **face_recognition** library by Adam Geitgey
- **MediaPipe** for facial landmark detection

## 📞 Support

For support and questions:

- **Email**: adityasuyal0001@gmail.com
- **Institution**: Birla Institute of Applied Sciences


---

<div align="center">





</div>



