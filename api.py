from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
import qrcode
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import json
import requests
from supabase_client import supabase

app = FastAPI(title="FaceMark Pro API", description="API for FaceMark Pro Attendance System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models
class Student(BaseModel):
    name: str
    email: str
    course: str
    roll_number: str

class AttendanceRecord(BaseModel):
    name: str
    date: str
    time: str
    method: str

class QRCodeRequest(BaseModel):
    student_name: str

class AIInsightsRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# Helper functions
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # Get all student data from Supabase
    response = supabase.table("students_data").select("*").execute()
    students_data = response.data
    
    for student in students_data:
        # Get the image path from the student data
        name = student['Name']
        image_path = f"Training_images/{name}.jpg"
        
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    
    return known_face_encodings, known_face_names

def mark_attendance(name, method="Face Recognition"):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")
    
    # Check if attendance already marked for today
    response = supabase.table("Attendance").select("*").eq("Name", name).eq("Date", date_string).execute()
    
    if not response.data:
        # Insert new attendance record
        data = {
            "Name": name,
            "Date": date_string,
            "Time": time_string,
            "Method": method
        }
        supabase.table("Attendance").insert(data).execute()
        return True
    return False

def generate_qr_code(student_name):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(student_name)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR code
    qr_code_path = f"QR_Codes/{student_name}_qr.png"
    img.save(qr_code_path)
    
    return qr_code_path

def generate_attendance_report(start_date=None, end_date=None):
    # Fetch attendance data from Supabase
    query = supabase.table("Attendance").select("*")
    
    if start_date:
        query = query.gte("Date", start_date)
    if end_date:
        query = query.lte("Date", end_date)
    
    response = query.execute()
    attendance_data = response.data
    
    if not attendance_data:
        return None
    
    # Create PDF report
    now = datetime.now()
    report_filename = f"Attendance_Report_{now.strftime('%Y-%m-%d')}.pdf"
    
    doc = SimpleDocTemplate(report_filename, pagesize=letter)
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    title = Paragraph("Attendance Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Create table data
    data = [["Name", "Date", "Time", "Method"]]
    for record in attendance_data:
        data.append([record["Name"], record["Date"], record["Time"], record["Method"]])
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    
    # Build PDF
    doc.build(elements)
    
    return report_filename

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

        # This would typically use an LLM API, but for now we'll return a simple analysis
        insights = {
            "total_records": total_records,
            "unique_attendance_dates": unique_attendance_dates,
            "student_data": student_attendance_data,
            "summary": "This is a placeholder for AI-generated insights. In production, this would connect to an LLM API."
        }
        
        return insights

    except Exception as e:
        return {"error": f"An unexpected error occurred during AI insights generation: {str(e)}"}

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to FaceMark Pro API"}

@app.post("/register-student")
async def register_student(student: Student, image: UploadFile = File(...)):
    try:
        # Save student image
        image_path = f"Training_images/{student.name}.jpg"
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        
        # Save student data to Supabase
        student_data = {
            "Name": student.name,
            "Email": student.email,
            "Course": student.course,
            "RollNumber": student.roll_number
        }
        
        supabase.table("students_data").insert(student_data).execute()
        
        # Generate QR code
        qr_code_path = generate_qr_code(student.name)
        
        return {"message": "Student registered successfully", "qr_code_path": qr_code_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/face-recognition")
async def recognize_face(image: UploadFile = File(...)):
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load known faces
        known_face_encodings, known_face_names = load_known_faces()
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        recognized_names = []
        
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                
                # Mark attendance
                is_new = mark_attendance(name)
                recognized_names.append({"name": name, "new_attendance": is_new})
        
        return {"recognized": recognized_names}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qr-code")
def create_qr_code(request: QRCodeRequest):
    try:
        qr_code_path = generate_qr_code(request.student_name)
        return FileResponse(qr_code_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan-qr")
async def scan_qr_code(image: UploadFile = File(...)):
    try:
        # Read QR code image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Decode QR code
        decoded_objects = decode(pil_img)
        
        if not decoded_objects:
            return {"message": "No QR code found"}
        
        # Get student name from QR code
        student_name = decoded_objects[0].data.decode("utf-8")
        
        # Mark attendance
        is_new = mark_attendance(student_name, method="QR Code")
        
        return {"student_name": student_name, "new_attendance": is_new}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance")
def get_attendance(start_date: Optional[str] = None, end_date: Optional[str] = None):
    try:
        query = supabase.table("Attendance").select("*")
        
        if start_date:
            query = query.gte("Date", start_date)
        if end_date:
            query = query.lte("Date", end_date)
        
        response = query.execute()
        attendance_data = response.data
        
        return {"attendance": attendance_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/students")
def get_students():
    try:
        response = supabase.table("students_data").select("*").execute()
        students_data = response.data
        
        return {"students": students_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
def create_report(start_date: Optional[str] = Form(None), end_date: Optional[str] = Form(None)):
    try:
        report_path = generate_attendance_report(start_date, end_date)
        
        if not report_path:
            return {"message": "No attendance data available for the specified period"}
        
        return FileResponse(report_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai-insights")
def get_ai_insights():
    try:
        insights = generate_ai_insights()
        return {"insights": insights}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance-stats")
def get_attendance_stats():
    try:
        # Fetch attendance data from Supabase
        response_attendance = supabase.table("Attendance").select("*").execute()
        attendance_data = response_attendance.data
        if not attendance_data:
            return {"message": "No attendance data available"}
        
        attendance_df = pd.DataFrame(attendance_data)
        
        # Fetch student data from Supabase
        response_students = supabase.table("students_data").select("*").execute()
        students_data = response_students.data
        if not students_data:
            return {"message": "No student data available"}
        
        students_df = pd.DataFrame(students_data)
        
        unique_attendance_dates = attendance_df['Date'].nunique()
        
        stats = []
        for index, student in students_df.iterrows():
            student_name = student['Name']
            student_attendance_records = attendance_df[attendance_df['Name'].str.upper() == student_name.upper()]
            attended_days = student_attendance_records['Date'].nunique()
            attendance_percentage = (attended_days / unique_attendance_dates * 100) if unique_attendance_dates > 0 else 0
            
            stats.append({
                "name": student_name,
                "attended_days": attended_days,
                "total_days": unique_attendance_dates,
                "percentage": round(attendance_percentage, 1)
            })
        
        return {"stats": stats}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)