import json
import os
from datetime import datetime
import streamlit as st
from supabase_client import supabase

OFFLINE_FILE = "offline_attendance.json"

def is_online():
    """Check if internet connection is available"""
    try:
        response = supabase.table("Attendance").select("*").limit(1).execute()
        return True
    except:
        return False

def save_offline_attendance(student_name, method="Face Recognition"):
    """Save attendance to local file when offline"""
    now = datetime.now()
    entry = {
        "Name": student_name.upper(),
        "Date": now.strftime('%Y-%m-%d'),
        "Time": now.strftime('%H:%M:%S'),
        "Method": method,
        "timestamp": now.isoformat()
    }
    
    # Load existing offline data
    offline_data = []
    if os.path.exists(OFFLINE_FILE):
        try:
            with open(OFFLINE_FILE, 'r') as f:
                offline_data = json.load(f)
        except:
            offline_data = []
    
    # Check for duplicates
    today = entry["Date"]
    existing = [r for r in offline_data if r["Name"] == entry["Name"] and r["Date"] == today]
    if existing:
        return False  # Already marked today
    
    # Add new entry
    offline_data.append(entry)
    
    # Save to file
    with open(OFFLINE_FILE, 'w') as f:
        json.dump(offline_data, f, indent=2)
    
    return True

def sync_offline_data():
    """Sync offline attendance data to Supabase"""
    if not os.path.exists(OFFLINE_FILE):
        return 0
    
    try:
        with open(OFFLINE_FILE, 'r') as f:
            offline_data = json.load(f)
        
        if not offline_data:
            return 0
        
        synced_count = 0
        remaining_data = []
        
        for entry in offline_data:
            try:
                # Check if already exists in Supabase
                existing = supabase.table("Attendance").select("*").eq("Name", entry["Name"]).eq("Date", entry["Date"]).execute()
                
                if not existing.data:
                    # Insert to Supabase
                    supabase.table("Attendance").insert({
                        "Name": entry["Name"],
                        "Date": entry["Date"],
                        "Time": entry["Time"],
                        "Method": entry["Method"]
                    }).execute()
                    synced_count += 1
                else:
                    # Already exists, don't keep in offline file
                    synced_count += 1
            except:
                # Failed to sync this entry, keep it for later
                remaining_data.append(entry)
        
        # Update offline file with remaining data
        with open(OFFLINE_FILE, 'w') as f:
            json.dump(remaining_data, f, indent=2)
        
        return synced_count
    
    except Exception as e:
        st.error(f"Sync error: {e}")
        return 0

def get_offline_count():
    """Get count of offline attendance records"""
    if not os.path.exists(OFFLINE_FILE):
        return 0
    
    try:
        with open(OFFLINE_FILE, 'r') as f:
            offline_data = json.load(f)
        return len(offline_data)
    except:
        return 0

def mark_attendance_offline_aware(student_name, method="Face Recognition"):
    """Mark attendance with offline support"""
    if is_online():
        # Try online first
        try:
            from main import mark_attendance_and_reward
            result = mark_attendance_and_reward(student_name, None)
            
            # Auto-sync any offline data when online
            synced = sync_offline_data()
            if synced > 0:
                st.success(f"✅ Synced {synced} offline records!")
            
            return result
        except:
            # Online failed, fall back to offline
            return save_offline_attendance(student_name, method)
    else:
        # Offline mode
        return save_offline_attendance(student_name, method)
