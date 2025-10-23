from supabase import create_client, Client

# WARNING: Hardcoding credentials is a security risk.
# It is recommended to use environment variables instead.
SUPABASE_URL = "https://ststvxhlxoiojrejxxaz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0c3R2eGhseG9pb2pyZWp4eGF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjExNzkxMjgsImV4cCI6MjA3Njc1NTEyOH0.E2hrFY8qjwM9ihSsh7SjCuhzock7nRzqU2yA2bUMbPQ"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error creating Supabase client: {e}")
    supabase = None
