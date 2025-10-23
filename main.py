import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import qrcode
import requests
from PIL import Image
from pyzbar.pyzbar import decode
import mediapipe as mp
from scipy.spatial import distance as dist
import json
import matplotlib.pyplot as plt
import plotly.express as px

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from supabase_client import supabase


# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot_visible" not in st.session_state:
    st.session_state.chatbot_visible = False


# Function: Generate AI Insights

def generate_ai_insights():
    try:
        # Fetch attendance data from Supabase
        response_attendance = supabase.table("Attendance").select("*").execute()
        attendance_data = response_attendance.data
        if not attendance_data:
            return "No attendance data available for analysis."
        attendance_df = pd.DataFrame(attendance_data)
        
        # Fetch student data from Supabase
        response_students = supabase.table("students_data").select("*").execute()
        students_data = response_students.data
        if not students_data:
            return "No student registration data available for comprehensive analysis."
        students_df = pd.DataFrame(students_data)
        
        total_records = len(attendance_df)
        unique_attendance_dates = attendance_df['Date'].nunique()
        
        
        student_attendance_data = []
        for index, student in students_df.iterrows():
            student_name = student['Name']
            student_attendance_records = attendance_df[attendance_df['Name'].str.upper() == student_name.upper()]
            attended_days = student_attendance_records['Date'].nunique()
            attendance_percentage = (attended_days / unique_attendance_dates * 100) if unique_attendance_dates > 0 else 0
            student_attendance_data.append(f"Student: {student_name}, Attended Days: {attended_days}, Total Days: {unique_attendance_dates}, Percentage: {attendance_percentage:.1f}%")

        prompt = f"""
        You are an AI assistant specialized in analyzing student attendance data.
        Below is a summary of attendance records:
        - Total attendance entries: {total_records}
        - Total unique class days: {unique_attendance_dates}

        Here is the attendance data for each student:
        {'; '.join(student_attendance_data)}

        Please provide a comprehensive and human-like analysis of the attendance data.
        Include insights for both consistently present students and students with lower attendance.
        Offer actionable recommendations to improve overall attendance and engagement.
        The analysis should be detailed, empathetic, and cover potential reasons for attendance patterns.
        The analysis should be detailed, empathetic, and cover potential reasons for attendance patterns.
        Aim for a narrative style, similar to a report from an educational consultant.
        """

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status() 
        
        llm_insights = response.json()["choices"][0]["message"]["content"]
        return llm_insights

    except Exception as e:
        return f"An unexpected error occurred during AI insights generation: {str(e)}"

def plot_attendance_comparison(selected_students=None):
    try:
        # Fetch attendance data from Supabase
        response_attendance = supabase.table("Attendance").select("*").execute()
        attendance_data = response_attendance.data
        if not attendance_data:
            st.warning("No attendance data available for plotting.")
            return None
        attendance_df = pd.DataFrame(attendance_data)
        
        # Fetch student data from Supabase
        response_students = supabase.table("students_data").select("*").execute()
        students_data = response_students.data
        if not students_data:
            st.warning("No student registration data available for plotting.")
            return None
        students_df = pd.DataFrame(students_data)
        
        unique_attendance_dates = attendance_df['Date'].nunique()
        
        plot_data = []
        for index, student in students_df.iterrows():
            student_name = student['Name']
            if selected_students and student_name not in selected_students:
                continue

            student_attendance_records = attendance_df[attendance_df['Name'].str.upper() == student_name.upper()]
            attended_days = student_attendance_records['Date'].nunique()
            
            attendance_percentage = (attended_days / unique_attendance_dates * 100) if unique_attendance_dates > 0 else 0
            plot_data.append({"Name": student_name, "Percentage": attendance_percentage})
        
        if not plot_data:
            st.info("No student data to plot for the selected criteria.")
            return None

        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df.sort_values(by="Percentage", ascending=False)

        fig = px.bar(plot_df, 
                     x="Percentage", 
                     y="Name", 
                     orientation='h',
                     title="Student Attendance Comparison",
                     labels={"Percentage": "Attendance Percentage (%)", "Name": "Student Name"},
                     color="Percentage",
                     color_continuous_scale=px.colors.sequential.Viridis)
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars by percentage
        fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
        
        return fig

    except Exception as e:
        st.error(f"An unexpected error occurred during plot generation: {str(e)}")
        return None


# Enhanced UI Configuration

st.set_page_config(
    page_title="FaceMark Pro - Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .attendance-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
    }
    
    .attendance-poor {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# AI Insights Configuration

GROQ_API_KEY = "yak"
GROQ_MODEL = "llama-3.3-70b-versatile"




SENDER_EMAIL = "adityasuyal0001@gmail.com"         # Replace with your Gmail address
SENDER_PASSWORD = "ibqw bipb petk lcvu"             # Replace with your Gmail App Password
ADMIN_EMAIL = "adityasuyal@birlainstitute.co.in"           # Replace with admin's email address


# Configuration

TRAINING_IMAGES_DIR = "Training_Images"             # Directory for training images
TOLERANCE = 0.6                                     # Face recognition matching threshold
MODEL = "hog"                                       # Use 'hog' for CPU and 'cnn' for GPU
qr_folder = "QR_Codes"                              # Folder for QR codes

# Liveness Detection Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
LIVENESS_INITIAL_WAIT = 3
LIVENESS_BLINK_CHALLENGE_DURATION = 3
LIVENESS_VERIFICATION_INTERVAL = 10
FACE_MOVEMENT_THRESHOLD = 10

os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)
os.makedirs(qr_folder, exist_ok=True)


# QR Code & Registration Functions

def upload_to_imgur(image_path):
    CLIENT_ID = "865c3e5bfc8ef5d"  # Your Imgur client ID
    headers = {"Authorization": f"Client-ID {CLIENT_ID}"}
    try:
        with open(image_path, "rb") as img:
            response = requests.post(
                "https://api.imgur.com/3/upload",
                headers=headers,
                files={"image": img},
                timeout=5  # prevent hanging if Imgur unreachable
            )
        if response.status_code == 200:
            return response.json()["data"]["link"]
        else:
            st.warning("⚠️ Imgur upload failed. Generating QR without image link.")
            return None
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Cannot connect to Imgur: {e}")
        return None


def generate_qr_with_image_url(name, img_path):
    image_url = upload_to_imgur(img_path)
    if image_url:
        qr_data = f"Name: {name}\nStudent at Birla Institute of Applied Sciences\nPhoto: {image_url}"
        qr = qrcode.QRCode(version=4, box_size=10, border=4)
        qr.add_data(qr_data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill="black", back_color="white")
        qr_path = f"{qr_folder}/{name}_qr.png"
        qr_img.save(qr_path)
        return qr_path, image_url
    return None, None

# Simple QR generation without Imgur dependency
def generate_simple_qr(name):
    qr_data = f"Name: {name}\nStudent at Birla Institute of Applied Sciences"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill="black", back_color="white")
    qr_path = f"{qr_folder}/{name}_qr.png"
    qr_img.save(qr_path)
    return qr_path

# Function to scan QR code for attendance
def scan_qr_code():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access camera.")
        return None
    
    st.write("📱 Scanning QR Code... Hold QR code in front of camera")
    image_placeholder = st.empty()
    
    if "stop_qr_scanner" not in st.session_state:
        st.session_state.stop_qr_scanner = False
    
    stop_button = st.button("Stop QR Scanner", key="stop_qr_scanner_button")
    
    while not st.session_state.stop_qr_scanner:
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(frame_rgb, channels="RGB")
        
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            qr_data = obj.data.decode("utf-8")
            # Extract name from QR data
            if "Name: " in qr_data:
                student_name = qr_data.split("Name: ")[1].split("\n")[0]
            else:
                student_name = qr_data
            cap.release()
            cv2.destroyAllWindows()
            return student_name
        
        if stop_button:
            st.session_state.stop_qr_scanner = True
    
    cap.release()
    cv2.destroyAllWindows()
    return None


# Cached Function: Load Known Faces

@st.cache_data
def get_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(TRAINING_IMAGES_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img_path = os.path.join(TRAINING_IMAGES_DIR, filename)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_faces.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])
    return known_faces, known_names


# Function: Recognize Face

def recognize_face(unknown_image):
    known_faces, known_names = get_known_faces()
    unknown_encoding = face_recognition.face_encodings(unknown_image)
    if unknown_encoding:
        matches = face_recognition.compare_faces(known_faces, unknown_encoding[0], TOLERANCE)
        if True in matches:
            matched_index = matches.index(True)
            return known_names[matched_index]
    return None


# Function: Update Rewards

def update_rewards(student_name):
    try:
        # Fetch existing reward data for the student from Supabase
        response = supabase.table("rewards").select("*").eq("Name", student_name).execute()
        
        if response.data:
            # Student exists, update attendance count
            current_reward = response.data[0]
            new_attendance_count = current_reward["AttendanceCount"] + 1
            
            def get_badge(count):
                if count >= 10:
                    return "Gold"
                elif count >= 5:
                    return "Silver"
                elif count >= 4:
                    return "Bronze"
                else:
                    return "No Badge"
            
            new_badge = get_badge(new_attendance_count)
            
            update_data = {
                "AttendanceCount": new_attendance_count,
                "Badge": new_badge
            }
            supabase.table("rewards").update(update_data).eq("Name", student_name).execute()
            
            return {"Name": student_name, "AttendanceCount": new_attendance_count, "Badge": new_badge}
        else:
            # Student does not exist, insert new record
            new_attendance_count = 1
            new_badge = "No Badge" # Initial badge
            
            new_entry_data = {
                "Name": student_name,
                "AttendanceCount": new_attendance_count,
                "Badge": new_badge
            }
            supabase.table("rewards").insert(new_entry_data).execute()
            return {"Name": student_name, "AttendanceCount": new_attendance_count, "Badge": new_badge}
            
    except Exception as e:
        st.error(f"Error updating rewards in Supabase: {e}")
        return {"Name": student_name, "AttendanceCount": 0, "Badge": "Error"}


# Function: Mark Attendance & Update Rewards, then Send Email

def mark_attendance_and_reward(student_name, frame):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')
    
    # Check if attendance for this student is already marked today in Supabase
    try:
        response = supabase.table("Attendance").select("*").eq("Name", student_name).eq("Date", dateString).execute()
        if response.data:
            existing_method = response.data[0].get("Method", "Unknown")
            return None, None, None, existing_method
    except Exception as e:
        print(f"Error checking existing attendance in Supabase: {e}")
        return None, None, None, "Error"

    # If not marked, insert new attendance record into Supabase
    try:
        new_entry_data = {
            "Name": student_name,
            "Date": dateString,
            "Time": timeString,
            "Method": "Face Recognition"
        }
        supabase.table("Attendance").insert(new_entry_data).execute()
    except Exception as e:
        print(f"Error inserting new attendance record into Supabase: {e}")
        return None, None, None, "Error"
    
    reward_info = update_rewards(student_name)
    send_email(student_name, frame, f"{dateString} {timeString}", reward_info['Badge'])
    return dateString, timeString, reward_info, "Face Recognition"

# Simple attendance marking for QR code scanning (ENHANCED)
def mark_attendance(student_name):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')
    
    # Check if attendance for this student is already marked today in Supabase
    try:
        response = supabase.table("Attendance").select("*").eq("Name", student_name).eq("Date", dateString).execute()
        if response.data:
            existing_method = response.data[0].get("Method", "Unknown")
            return False, existing_method
    except Exception as e:
        print(f"Error checking existing attendance in Supabase: {e}")
        return False, "Error"

    # If not marked, insert new attendance record into Supabase
    try:
        new_entry_data = {
            "Name": student_name,
            "Date": dateString,
            "Time": timeString,
            "Method": "QR Code"
        }
        supabase.table("Attendance").insert(new_entry_data).execute()
    except Exception as e:
        print(f"Error inserting new attendance record into Supabase: {e}")
        return False, "Error"
    
    update_rewards(student_name)
    return True, "QR Code"

# -------------------------
# Function: Send Email Notification with Image and Timestamp
# -------------------------
def send_email(student_name, captured_img, timestamp, badge):
    subject = "Attendance Marked Notification"
    body = f"Attendance for {student_name} has been marked at {timestamp}.\nCurrent Badge: {badge}"
    
    msg = MIMEMultipart()
    msg["From"] = "GateAttend <{}>".format(SENDER_EMAIL)
    msg["To"] = ADMIN_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    retval, buffer = cv2.imencode('.jpg', captured_img)
    if retval:
        image_bytes = buffer.tobytes()
        image_mime = MIMEImage(image_bytes, name="attendance.jpg")
        msg.attach(image_mime)
    else:
        st.error("❌ Failed to encode image for attachment.")
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
        server.quit()
        st.success(f"✅ Email sent to Admin@BIAS for {student_name} at {timestamp}!")
    except Exception as e:
        st.error(f"❌ Error sending email: {e}")

# -------------------------
# Function: Send Parent Notifications
# -------------------------
def send_parent_notification(student_name, parent_email, notification_type, attendance_percentage=None):
    subject = f"Low Attendance Alert - {student_name}"
    body = f"""
Dear Parent/Guardian,

We would like to inform you that {student_name}'s attendance has dropped to {attendance_percentage}%, 
which is below the required 75% threshold.

Current attendance: {attendance_percentage}%
Required minimum: 75%

We encourage you to discuss the importance of regular attendance with your child.

Best regards,
Birla Institute of Applied Sciences
    """
    
    msg = MIMEMultipart()
    msg["From"] = f"Attendance Alert <{SENDER_EMAIL}>"
    msg["To"] = parent_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, parent_email, msg.as_string())
        server.quit()
        st.success(f"✅ Email sent to {parent_email} for {student_name} ({attendance_percentage}% attendance)!")
    except Exception as e:
        st.error(f"❌ Error sending email to {parent_email}: {e}")

def check_and_notify_low_attendance():
    try:
        response_students = supabase.table("students_data").select("*").execute()
        students_data = response_students.data
        if not students_data:
            st.warning("⚠️ No student data found.")
            return
        students_df = pd.DataFrame(students_data)

        response_attendance = supabase.table("Attendance").select("*").execute()
        attendance_data = response_attendance.data
        if not attendance_data:
            st.warning("⚠️ No attendance records found.")
            return
        attendance_df = pd.DataFrame(attendance_data)
        
        total_classes = len(attendance_df["Date"].unique())
        defaulters = []
        
        for _, student in students_df.iterrows():
            student_name = student["Name"]
            parent_email = student.get("Parent_Gmail", "")
            parent_name = student.get("Parent_Name", "")
            
            if not parent_email:
                continue
                
            student_attendance = attendance_df[attendance_df["Name"].str.upper() == student_name.upper()]
            attended_classes = len(student_attendance["Date"].unique())
            percentage = (attended_classes / total_classes * 100) if total_classes > 0 else 0
            
            if percentage < 75:
                defaulters.append({
                    "Name": student_name,
                    "Percentage": round(percentage, 2),
                    "Parent_Email": parent_email,
                    "Parent_Name": parent_name
                })
        
        if not defaulters:
            st.success("✅ No students with low attendance found!")
            return
        
        st.subheader("⚠️ Students with Low Attendance (<75%)")
        st.write("Sending emails to parents of students with low attendance...")
        
        for defaulter in defaulters:
            st.write(f"• {defaulter['Name']} - {defaulter['Percentage']}% "
                     f"(Parent: {defaulter['Parent_Name']} - {defaulter['Parent_Email']})")
            try:
                subject = f"Low Attendance Alert - {defaulter['Name']}"
                body = f'''Dear Parent/Guardian,

We would like to inform you that {defaulter['Name']}'s attendance has dropped to {defaulter['Percentage']}%, which is below the required 75% threshold.

Current attendance: {defaulter['Percentage']}% 
Required minimum: 75%

We encourage you to discuss the importance of regular attendance with your child.

Best regards,
Birla Institute of Applied Sciences'''
                
                msg = MIMEMultipart()
                msg["From"] = f"Attendance Alert <{SENDER_EMAIL}>"
                msg["To"] = defaulter['Parent_Email']
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))

                # Find and attach student image
                image_path = None
                for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                    path = os.path.join(TRAINING_IMAGES_DIR, f"{defaulter['Name']}{ext}")
                    if os.path.exists(path):
                        image_path = path
                        break
                
                if image_path:
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
                
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, defaulter['Parent_Email'], msg.as_string())
                server.quit()
                st.success(f"✅ Email sent to {defaulter['Parent_Email']} for {defaulter['Name']}!")
            except Exception as e:
                st.error(f"❌ Error sending email to {defaulter['Parent_Email']}: {e}")

    except Exception as e:
        st.error(f"❌ Error checking attendance: {e}")

# -------------------------
# Function: Calculate Attendance Percentage
# -------------------------
def calculate_attendance_percentage():
    try:
        response_students = supabase.table("students_data").select("*").execute()
        students_data = response_students.data
        if not students_data:
            st.warning("⚠️ No student data found.")
            return None
        students_df = pd.DataFrame(students_data)

        response_attendance = supabase.table("Attendance").select("*").execute()
        attendance_data = response_attendance.data
        if not attendance_data:
            st.warning("⚠️ No attendance records found.")
            return None
        attendance_df = pd.DataFrame(attendance_data)
        
        # Get total unique dates (total classes)
        total_classes = len(attendance_df["Date"].unique())
        
        # Calculate attendance for each student
        attendance_summary = []
        
        for _, student in students_df.iterrows():
            student_name = student["Name"]
            
            # Count attendance for this student (case-insensitive)
            student_attendance = attendance_df[attendance_df["Name"].str.upper() == student_name.upper()]
            attended_classes = len(student_attendance["Date"].unique())
            
            # Calculate percentage
            percentage = (attended_classes / total_classes * 100) if total_classes > 0 else 0
            
            attendance_summary.append({
                "Name": student_name,
                "Attended": attended_classes,
                "Total": total_classes,
                "Percentage": round(percentage, 2)
            })
        
        return pd.DataFrame(attendance_summary)
    
    except Exception as e:
        st.error(f"❌ Error calculating attendance: {e}")
        return None


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -------------------------
# Function: Generate Attendance Report (Optional)
# -------------------------
def generate_attendance_report():
    response = supabase.table("Attendance").select("*").execute()
    attendance_data = response.data
    if not attendance_data:
        st.warning("⚠️ No attendance records found.")
        return
    df = pd.DataFrame(attendance_data)

    if df.empty:
        st.warning("⚠️ No attendance records found.")
        return
    unique_dates = df["Date"].unique()
    for date in unique_dates:
        date_df = df[df["Date"] == date]
        filename = f"Attendance_Report_{date}.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        c.drawString(100, 750, f"Attendance Report - {date}")
        y_position = 720
        for index, row in date_df.iterrows():
            c.drawString(100, y_position, f"{row['Name']} - {row['Time']}")
            y_position -= 20
        c.save()
        st.success(f"✅ Attendance Report Generated for {date}: {filename}")

# Function: Chatbot Response
def chatbot_response(user_query):
    try:
        # Fetch attendance data from Supabase
        response_attendance = supabase.table("Attendance").select("*").execute()
        attendance_data = response_attendance.data
        if not attendance_data:
            return "I'm sorry, but there is no attendance data available right now. How else can I assist you?"
        attendance_df = pd.DataFrame(attendance_data)
        
        # Fetch student data from Supabase
        response_students = supabase.table("students_data").select("*").execute()
        students_data = response_students.data
        if not students_data:
            return "I couldn't find any student registration data. Would you like to register a new student?"
        students_df = pd.DataFrame(students_data)
        
        # Prepare a concise prompt for the LLM
        prompt = f"""
        You are an AI assistant for a student attendance system. Respond concisely and directly to the user's query.
        If appropriate, suggest follow-up questions or actions the user can take.

        Attendance Data: {attendance_df.to_dict(orient='records')}
        Student Data: {students_df.to_dict(orient='records')}

        User Query: {user_query}

        Provide a short, helpful response and suggest follow-up questions or actions.
        """
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5  # Lower temperature for more focused responses
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        llm_response = response.json()["choices"][0]["message"]["content"]
        return llm_response.strip()

    except Exception as e:
        return f"An error occurred while processing your query: {str(e)}. How else can I assist you?"

# -------------------------
# Enhanced Streamlit UI
# -------------------------

# Main Header with Gradient Background
st.markdown("""
<div class="main-header">
    <h1>FaceMark Pro</h1>
    <h3>Advanced Attendance & Absence Management System</h3>
    <p>Facial Recognition • QR Code Scanning • Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)

# Dashboard Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_records = 0
    if supabase is None:
        st.error("Supabase client is not initialized for Total Records metric.")
    else:
        try:
            # Fetch the total count of attendance records
            response_attendance = supabase.table("Attendance").select("*").execute()
            total_records = len(response_attendance.data) if response_attendance.data else 0  # Fix: Use len() to count records
        except Exception as e:
            st.error(f"Supabase Error (Total Records metric): {e}")
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 {total_records}</h3>
        <p>Total Records</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    today_records = 0
    if supabase is None:
        st.error("Supabase client is not initialized for Today's Attendance metric.")
    else:
        try:
            response_attendance = supabase.table("Attendance").select("*").eq("Date", datetime.now().strftime('%Y-%m-%d')).execute()
            today_records = len(response_attendance.data) if response_attendance.data else 0
        except Exception as e:
            st.error(f"Supabase Error (Today's Attendance metric): {e}")
    st.markdown(f"""
    <div class="metric-card">
        <h3>📅 {today_records}</h3>
        <p>Today's Attendance</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    registered_faces = 0
    if supabase is None:
        st.error("Supabase client is not initialized for Registered Students metric.")
    else:
        try:
            # Fetch all registered students
            response_students = supabase.table("students_data").select("*").execute()
            registered_faces = len(response_students.data) if response_students.data else 0  # Fix: Use len() to count records
        except Exception as e:
            st.error(f"Supabase Error (Registered Students metric): {e}")
    st.markdown(f"""
    <div class="metric-card">
        <h3>👥 {registered_faces}</h3>
        <p>Registered Students</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    response_rewards = supabase.table("rewards").select("*").eq("Badge", "Gold").execute()
    gold_badges = len(response_rewards.data) if response_rewards.data else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔒 {gold_badges}</h3>
        <p>Anti-Spoof Active</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Sidebar with Better Styling
st.sidebar.markdown("""
<div style="background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; text-align: center;">🎛️ Control Panel</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📋 Reports & Analytics")
if st.sidebar.button("📊 Download Attendance Report", key="download_report"):
    generate_attendance_report()

if st.sidebar.button("⚠️ Send Low Attendance Notice", key="check_attendance"):
    check_and_notify_low_attendance()

if st.sidebar.button("🤖 AI Attendance Insights", key="ai_insights"):
    st.markdown("""
    <div class="info-card">
        <h3>🤖 AI-Powered Attendance Analysis</h3>
        <p>Generating intelligent insights from your attendance data...</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🧠 AI is analyzing attendance patterns..."):
        insights = generate_ai_insights()
    
    st.markdown(f"""
    <div class="success-card">
        <h4>📊 AI Insights & Recommendations</h4>
        <p>{insights}</p>
    </div>
    """, unsafe_allow_html=True)

    # Get all student names for multiselect
    all_student_names = []
    response_students = supabase.table("students_data").select("Name").execute()
    if response_students.data:
        students_df = pd.DataFrame(response_students.data)
        all_student_names = students_df['Name'].tolist()

    selected_students = st.multiselect(
        "Select students to compare in the graph (leave empty for all):",
        options=all_student_names,
        key="student_selector"
    )

    # Plot the attendance comparison graph
    st.subheader("📈 Attendance Comparison Graph")
    attendance_plot = plot_attendance_comparison(selected_students)
    if attendance_plot:
        st.plotly_chart(attendance_plot)
    else:
        st.info("No graph generated due to insufficient data or selection.")

st.sidebar.markdown("### 📹 Camera Controls")
if st.sidebar.button("🎥 Start Webcam", key="start_webcam"):
    st.session_state["webcam_active"] = True
if st.sidebar.button("⏹️ Stop Webcam", key="stop_webcam"):
    st.session_state["webcam_active"] = False

st.sidebar.markdown("### 👤 Registration")
if st.sidebar.button("📸 Register New Face", key="register_face"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Unable to access the webcam.")
    else:
        with st.spinner("📸 Capturing Image... Please wait for 4 seconds."):
            start_time = time.time()
            last_frame = None
            while time.time() - start_time < 4:
                success, frame = cap.read()
                if success:
                    last_frame = frame
        cap.release()
        if last_frame is not None:
            st.session_state["captured_image"] = last_frame
            st.markdown("""
            <div class="success-card">
                <h4>✅ Image Captured Successfully!</h4>
                <p>Please fill the registration form below.</p>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.markdown("### 📱 QR Code Scanner")
if st.sidebar.button("📱 Scan QR for Attendance", key="scan_qr"):
    st.markdown("""
    <div class="info-card">
        <h4>📱 QR Code Attendance Scanner</h4>
        <p>Hold your QR code in front of the camera</p>
    </div>
    """, unsafe_allow_html=True)
    
    student_name = scan_qr_code()
    if student_name:
        result, method = mark_attendance(student_name)
        if result:
            st.markdown(f"""
            <div class="success-card">
                <h4>✅ Attendance Marked Successfully!</h4>
                <p><strong>{student_name}</strong> marked via QR Code</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="warning-card">
                <h4>⚠️ Already Marked</h4>
                <p><strong>{student_name}</strong> already marked with {method}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ No valid QR code detected.")

st.sidebar.markdown("### 📊 View Data")
if st.sidebar.button("📋 Show Attendance Records", key="show_attendance"):
    response = supabase.table("Attendance").select("*").execute()
    attendance_data = response.data
    if attendance_data:
        df = pd.DataFrame(attendance_data)
        st.markdown("### 📋 Attendance Records")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No attendance records found.")

if st.sidebar.button("📈 Show Attendance Percentage", key="show_percentage"):
    attendance_summary = calculate_attendance_percentage()
    if attendance_summary is not None and not attendance_summary.empty:
        st.markdown("### 📊 Student Attendance Analysis")
        
        # Separate students by attendance percentage
        good_attendance = attendance_summary[attendance_summary["Percentage"] >= 75]
        poor_attendance = attendance_summary[attendance_summary["Percentage"] < 75]
        
        # Display students with good attendance (≥75%)
        if not good_attendance.empty:
            st.markdown("#### ✅ Students with Good Attendance (≥75%)")
            for _, student in good_attendance.iterrows():
                st.markdown(f"""
                <div class="attendance-good">
                    <strong>{student['Name']}</strong>: {student['Percentage']}% ({student['Attended']}/{student['Total']})
                </div>
                """, unsafe_allow_html=True)
        
        # Display students with poor attendance (<75%)
        if not poor_attendance.empty:
            st.markdown("#### ⚠️ Students with Poor Attendance (<75%)")
            for _, student in poor_attendance.iterrows():
                st.markdown(f"""
                <div class="attendance-poor">
                    <strong>{student['Name']}</strong>: {student['Percentage']}% ({student['Attended']}/{student['Total']})
                </div>
                """, unsafe_allow_html=True)
        
        # Display summary table
        st.markdown("#### 📋 Complete Summary")
        st.dataframe(attendance_summary, use_container_width=True)

# Sidebar Chatbot Toggle
st.sidebar.markdown("### 🤖 Chatbot")
if st.sidebar.button("💬 Toggle Chatbot", key="toggle_chatbot"):
    st.session_state.chatbot_visible = not st.session_state.chatbot_visible

# Chatbot UI
if st.session_state.chatbot_visible:
    st.markdown("""
    <div class="info-card">
        <h3>🤖 AI Chatbot</h3>
        <p>Ask me anything about student attendance or project insights!</p>
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask me anything about student attendance or project insights..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chatbot_response(user_input)
                    if response:
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        st.error("Chatbot returned an empty response.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.write(f"Debug: {e}")

# Initialize webcam control, last recognized face, and last marked time in session state
if "webcam_active" not in st.session_state:
    st.session_state["webcam_active"] = False
if "last_recognized" not in st.session_state:
    st.session_state["last_recognized"] = None
if "last_marked_time" not in st.session_state:
    st.session_state["last_marked_time"] = 0
if "last_face_location" not in st.session_state:
    st.session_state["last_face_location"] = None

# Enhanced Registration Form
if "captured_image" in st.session_state:
    st.markdown("""
    <div class="info-card">
        <h3>📝 Student Registration Form</h3>
        <p>Please fill in all the details below to complete registration</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("🆔 Student ID:", placeholder="Enter student ID")
        name = st.text_input("👤 Full Name:", placeholder="Enter full name")
        age = st.number_input("🎂 Age:", min_value=1, max_value=100, value=18)
        grade = st.text_input("🎓 Grade/Class:", placeholder="Enter grade or class")
    
    with col2:
        parent_name = st.text_input("👨‍👩‍👧‍👦 Parent/Guardian Name:", placeholder="Enter parent name")
        parent_contact = st.text_input("📞 Parent Contact:", placeholder="Enter contact number")
        parent_gmail = st.text_input("📧 Parent Gmail:", placeholder="Enter email address")
    
    def handle_student_registration(student_id, name, age, grade, parent_name, parent_contact, parent_gmail, captured_image):
        if all([student_id, name, grade, parent_name, parent_contact, parent_gmail]):
            with st.spinner("🔄 Processing registration..."):
                # Check if face already exists
                recognized_name = recognize_face(captured_image)
                
                if recognized_name:
                    st.markdown(f"""
                    <div class="warning-card">
                        <h4>⚠️ Face Already Registered!</h4>
                        <p>This face is already registered as: <strong>{recognized_name}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Save student data to Supabase
                    student_data = {
                        "Student_ID": student_id,
                        "Name": name,
                        "Age": age,
                        "Grade": grade,
                        "Parent_Name": parent_name,
                        "Parent_Contact": parent_contact,
                        "Parent_Gmail": parent_gmail
                    }
                    
                    try:
                        supabase.table("students_data").insert(student_data).execute()
                    except Exception as e:
                        st.error(f"Error saving student data to Supabase: {e}")
                        return # Stop further processing if save fails
                    
                    # Save face image
                    file_path = f"{TRAINING_IMAGES_DIR}/{name}.jpg"
                    cv2.imwrite(file_path, captured_image)

                    # Generate QR code
                    qr_path, image_url = generate_qr_with_image_url(name, file_path)
                    if qr_path:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(qr_path, caption="Generated QR Code", width=300)
                        with col2:
                            st.markdown(f"""
                            <div class="success-card">
                                <h4>✅ Registration Completed!</h4>
                                <p><strong>Student:</strong> {name}</p>
                                <p><strong>ID:</strong> {student_id}</p>
                                <p><strong>Grade:</strong> {grade}</p>
                                <p><strong>QR Code:</strong> Generated Successfully</p>
                            </div>
                            """, unsafe_allow_html=True)
                        if image_url:
                            st.info(f"📷 **Student Image URL:** [{image_url}]({image_url})")
                    else:
                        # Fallback to simple QR generation
                        qr_path = generate_simple_qr(name)
                        st.image(qr_path, caption="Generated QR Code", width=300)
                        st.markdown("""
                        <div class="success-card">
                            <h4>✅ Registration Completed!</h4>
                            <p>QR Code generated successfully</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Update the UI dynamically after registration
                    st.experimental_rerun()  # Refresh the app to reflect the new count

                    del st.session_state["captured_image"]
                    st.balloons()
        else:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ Incomplete Information</h4>
                <p>Please fill in all fields before registering.</p>
            </div>
            """, unsafe_allow_html=True)

if st.button("✅ Register Student", key="register_student"):
    handle_student_registration(student_id, name, age, grade, parent_name, parent_contact, parent_gmail, st.session_state["captured_image"])

# -------------------------
# Enhanced Live Webcam Capture with Anti-Spoof Detection
# -------------------------
if st.session_state.get("webcam_active"):
    st.markdown("""
    <div class="info-card">
        <h3>🎥 Live Face Recognition with Anti-Spoof Detection</h3>
        <p>Position yourself in front of the camera and blink naturally for verification</p>
    </div>
    """, unsafe_allow_html=True)
    
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        status_placeholder = st.empty()
    with col2:
        recognition_placeholder = st.empty()
    with col3:
        liveness_placeholder = st.empty()
    
    # Liveness detection setup
    EYE_AR_THRESH = 0.25 # Adjusted for better blink detection
    EYE_AR_CONSEC_FRAMES = 2 # Adjusted for faster blink detection
    if "blink_counter" not in st.session_state:
        st.session_state.blink_counter = 0
    
    if "liveness_state" not in st.session_state:
        st.session_state.liveness_state = "IDLE" # States: "IDLE", "WAITING_INITIAL", "CHALLENGE_BLINK", "VERIFIED"
    if "liveness_state_start_time" not in st.session_state:
        st.session_state.liveness_state_start_time = 0 # Timestamp when the current liveness state began
    if "liveness_verified_until" not in st.session_state:
        st.session_state.liveness_verified_until = 0 # Timestamp when liveness verification expires
    
    LIVENESS_INITIAL_WAIT = 3 # seconds
    LIVENESS_BLINK_CHALLENGE_DURATION = 3 # seconds
    LIVENESS_VERIFICATION_INTERVAL = 10 # seconds, how long liveness remains verified

    while st.session_state.get("webcam_active"):
        success, frame = cap.read()
        if not success:
            st.error("⚠️ Failed to capture image from webcam!")
            break

        current_time = time.time()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        
        if face_locations:
            status_placeholder.markdown("🟢 **Live Face Detected**")
            
            current_face_location = face_locations[0] # Assuming one face
            
            # Check for face stability
            if st.session_state.last_face_location is not None:
                # Calculate center of current face
                cy1, cx2, cy2, cx1 = current_face_location
                current_face_center_x = (cx1 + cx2) / 2
                current_face_center_y = (cy1 + cy2) / 2

                # Calculate center of last face
                ly1, lx2, ly2, lx1 = st.session_state.last_face_location
                last_face_center_x = (lx1 + lx2) / 2
                last_face_center_y = (ly1 + ly2) / 2

                movement_distance = dist.euclidean(
                    (current_face_center_x, current_face_center_y),
                    (last_face_center_x, last_face_center_y)
                )

                if movement_distance > FACE_MOVEMENT_THRESHOLD:
                    # Face moved too much, reset liveness state
                    st.session_state.liveness_state = "IDLE"
                    st.session_state.liveness_state_start_time = 0
                    st.session_state.liveness_verified_until = 0
                    st.session_state.blink_counter = 0
                    liveness_placeholder.markdown("⚠️ **Unstable face detected. Resetting liveness.**")
                    last_face_location = current_face_location # Update for next frame
                    FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
                    time.sleep(0.03)
                    continue # Skip further processing for this frame
            
            st.session_state.last_face_location = current_face_location # Update for next frame
            
            # Check if liveness is currently verified
            if current_time < st.session_state.liveness_verified_until:
                st.session_state.liveness_state = "VERIFIED"
                liveness_placeholder.markdown("✅ **Liveness Verified**")
                # Proceed with face recognition and attendance marking
                encodesCurFrame = face_recognition.face_encodings(rgb_small_frame, face_locations)
                known_faces, known_names = get_known_faces()

                for encodeFace, faceLoc in zip(encodesCurFrame, face_locations):
                    matches = face_recognition.compare_faces(known_faces, encodeFace, TOLERANCE)
                    faceDis = face_recognition.face_distance(known_faces, encodeFace)
                    matchIndex = np.argmin(faceDis) if len(faceDis) > 0 else None
                    threshold = 0.5

                    if matchIndex is not None and matches[matchIndex] and faceDis[matchIndex] < threshold:
                        current_name = known_names[matchIndex].upper()
                        recognition_placeholder.markdown(f"✅ **Verified: {current_name}**")
                        
                        result = mark_attendance_and_reward(current_name, frame)
                        if result[0] is not None:
                            st.session_state["last_recognized"] = current_name
                            st.session_state["last_marked_time"] = time.time()
                            liveness_placeholder.markdown("✅ **Attendance Marked!**")
                            st.balloons()
                            st.session_state.liveness_verified_until = 0 # Reset to force re-verification for next attendance
                        elif len(result) > 3:
                            cv2.putText(frame, f"Already marked with {result[3]}", (50, 100),
                                       cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 165, 255), 2)

                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 3)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, f"{current_name} - VERIFIED", (x1 + 6, y2 - 6),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        recognition_placeholder.markdown("❓ **Unknown Face**")
            else:
                # Liveness not verified or expired, manage challenge states
                if st.session_state.liveness_state == "IDLE" or st.session_state.liveness_state == "VERIFIED":
                    st.session_state.liveness_state = "WAITING_INITIAL"
                    st.session_state.liveness_state_start_time = current_time
                    st.session_state.blink_counter = 0 # Reset blink counter for new challenge
                
                if st.session_state.liveness_state == "WAITING_INITIAL":
                    elapsed_initial_wait = current_time - st.session_state.liveness_state_start_time
                    if elapsed_initial_wait < LIVENESS_INITIAL_WAIT:
                        liveness_placeholder.markdown(f"⏳ **Initial wait: {LIVENESS_INITIAL_WAIT - elapsed_initial_wait:.1f}s**")
                    else:
                        st.session_state.liveness_state = "CHALLENGE_BLINK"
                        st.session_state.liveness_state_start_time = current_time # Reset timer for blink challenge
                        st.session_state.blink_counter = 0 # Reset blink counter for new challenge
                
                if st.session_state.liveness_state == "CHALLENGE_BLINK":
                    elapsed_blink_challenge = current_time - st.session_state.liveness_state_start_time
                    if elapsed_blink_challenge < LIVENESS_BLINK_CHALLENGE_DURATION:
                        liveness_placeholder.markdown(f"⏳ **Blink to verify ({LIVENESS_BLINK_CHALLENGE_DURATION - elapsed_blink_challenge:.1f}s remaining)**")
                        
                        if face_landmarks_list: # Ensure landmarks are available for blink check
                            face_landmarks = face_landmarks_list[0]
                            left_eye = face_landmarks['left_eye']
                            right_eye = face_landmarks['right_eye']
                            
                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            ear = (left_ear + right_ear) / 2.0
                            
                            if ear < EYE_AR_THRESH:
                                st.session_state.blink_counter += 1
                            else:
                                if st.session_state.blink_counter >= EYE_AR_CONSEC_FRAMES:
                                    # Blink detected within challenge duration!
                                    st.session_state.liveness_verified_until = current_time + LIVENESS_VERIFICATION_INTERVAL
                                    st.session_state.liveness_state = "VERIFIED" # Transition to verified state
                                    liveness_placeholder.markdown("✅ **Liveness Verified!**")
                                st.session_state.blink_counter = 0
                        else:
                            liveness_placeholder.markdown("⚠️ **No landmarks for blink check**")
                    else:
                        # Blink challenge duration expired, no blink detected
                        liveness_placeholder.markdown("❌ **Can't find human! Please try again.**")
                        st.session_state.liveness_state = "IDLE" # Reset to IDLE to restart the whole process
                        st.session_state.blink_counter = 0 # Reset blink counter
        else:
            status_placeholder.markdown("🔴 **No Face Detected**")
            st.session_state.liveness_state = "IDLE" # Reset state
            st.session_state.liveness_state_start_time = 0
            st.session_state.liveness_verified_until = 0
            blink_counter = 0
            last_face_location = None # Reset last face location if no face is detected

        FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.03)
    
    cap.release()
    status_placeholder.empty()
    recognition_placeholder.empty()
    liveness_placeholder.empty()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-top: 2rem;">
    <h4>🎓 FaceMark Pro - Attendance Management System</h4>
    <p>🔒 Anti-Spoof Protection • Powered by AI • Built with ❤️ for Educational Excellence</p>
    <p><small>© 2024 Birla Institute of Applied Sciences</small></p>
</div>
""", unsafe_allow_html=True)
