from web3 import Web3
import json
from datetime import datetime
import hashlib

class BlockchainAttendance:
    def __init__(self):
        # Connect to Ganache local blockchain (free)
        self.w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        self.account = self.w3.eth.accounts[0]
        
    def create_attendance_hash(self, student_id, timestamp, present, face_confidence=None):
        """Create tamper-proof attendance record"""
        data = f"{student_id}_{timestamp}_{present}_{face_confidence}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def store_attendance(self, student_id, present=True, face_confidence=None):
        """Store attendance on blockchain"""
        timestamp = datetime.now().isoformat()
        attendance_hash = self.create_attendance_hash(student_id, timestamp, present, face_confidence)
        
        # Store hash on blockchain
        tx_hash = self.w3.eth.send_transaction({
            'from': self.account,
            'to': self.account,
            'value': 0,
            'data': self.w3.to_hex(text=attendance_hash)
        })
        
        return {
            'student_id': student_id,
            'timestamp': timestamp,
            'present': present,
            'blockchain_hash': attendance_hash,
            'transaction_hash': tx_hash.hex()
        }
    
    def verify_attendance(self, student_id, timestamp, present, blockchain_hash):
        """Verify attendance record hasn't been tampered with"""
        expected_hash = self.create_attendance_hash(student_id, timestamp, present)
        return expected_hash == blockchain_hash

# Integration with existing attendance system
def mark_attendance_with_blockchain(student_id, face_confidence=None):
    blockchain = BlockchainAttendance()
    
    # Mark attendance on blockchain
    record = blockchain.store_attendance(student_id, True, face_confidence)
    
    # Continue with existing SMS/email notifications
    print(f"Attendance recorded on blockchain: {record['blockchain_hash'][:16]}...")
    return record

if __name__ == "__main__":
    # Test the system
    record = mark_attendance_with_blockchain("STU001", 0.95)
    print(json.dumps(record, indent=2))
