# Enhanced attendance functions with method tracking
import pandas as pd
from datetime import datetime
import os

ATTENDANCE_FILE = "Attendance.csv"

def mark_attendance_and_reward_enhanced(student_name, frame, send_email_func, update_rewards_func):
    """Enhanced function for face recognition attendance with method tracking"""
    # Load or create attendance CSV
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Method"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    else:
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if "Method" not in df.columns:
                df["Method"] = "Face Recognition"  # Default for existing records
                df.to_csv(ATTENDANCE_FILE, index=False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Method"])
    
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')
    
    # Check if attendance for this student is already marked today
    existing_record = df[(df["Name"] == student_name) & (df["Date"] == dateString)]
    if not existing_record.empty:
        existing_method = existing_record.iloc[0]["Method"] if "Method" in existing_record.columns else "Unknown"
        return None, None, None, existing_method
    else:
        new_entry = pd.DataFrame([[student_name, dateString, timeString, "Face Recognition"]], 
                                columns=["Name", "Date", "Time", "Method"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
    
        reward_info = update_rewards_func(student_name)
        send_email_func(student_name, frame, f"{dateString} {timeString}", reward_info['Badge'])
        return dateString, timeString, reward_info, "Face Recognition"

def mark_attendance_qr_enhanced(student_name, update_rewards_func):
    """Enhanced function for QR code attendance with method tracking"""
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Method"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    else:
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if "Method" not in df.columns:
                df["Method"] = "QR Code"  # Default for existing records
                df.to_csv(ATTENDANCE_FILE, index=False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Method"])
    
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')
    
    # Check if attendance already marked today
    existing_record = df[(df["Name"] == student_name) & (df["Date"] == dateString)]
    if not existing_record.empty:
        existing_method = existing_record.iloc[0]["Method"] if "Method" in existing_record.columns else "Unknown"
        return False, existing_method
    else:
        new_entry = pd.DataFrame([[student_name, dateString, timeString, "QR Code"]], 
                                columns=["Name", "Date", "Time", "Method"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        update_rewards_func(student_name)
        return True, "QR Code"
