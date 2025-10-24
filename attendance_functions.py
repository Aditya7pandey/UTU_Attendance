# Enhanced attendance functions with method tracking
import pandas as pd
from datetime import datetime
import os
import streamlit as st
from supabase_client import supabase

def mark_attendance_and_reward_enhanced(student_name, frame, send_email_func, update_rewards_func):
    """Enhanced function for face recognition attendance with method tracking"""
    if supabase is None:
        st.error("Supabase client is not initialized. Please check your Supabase URL and API Key.")
        return None, None, None, "Error"

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
        st.error(f"Supabase Error (checking existing attendance): {e}")
        # Fallback or error handling if Supabase query fails
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
    
    reward_info = update_rewards_func(student_name)
    send_email_func(student_name, frame, f"{dateString} {timeString}", reward_info['Badge'])
    return dateString, timeString, reward_info, "Face Recognition"

def mark_attendance_qr_enhanced(student_name, update_rewards_func):
    """Enhanced function for QR code attendance with method tracking"""
    if supabase is None:
        st.error("Supabase client is not initialized. Please check your Supabase URL and API Key.")
        return False, "Error"

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
        st.error(f"Supabase Error (checking existing attendance): {e}")
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
    
    update_rewards_func(student_name)
    return True, "QR Code"
