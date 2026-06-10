#!/usr/bin/env python3
"""
Quick blockchain integration checker for FaceMark Pro
"""

def check_blockchain_integration():
    print("🔍 Checking FaceMark Pro Blockchain Integration...")
    print("-" * 50)
    
    # Check 1: Import test
    try:
        from blockchain_attendance import BlockchainAttendance
        print("✅ Blockchain module imported successfully")
    except ImportError as e:
        print(f"❌ Blockchain module not found: {e}")
        return False
    
    # Check 2: Connection test
    try:
        blockchain = BlockchainAttendance()
        print("✅ Blockchain connection established")
    except Exception as e:
        print(f"❌ Blockchain connection failed: {e}")
        return False
    
    # Check 3: Functionality test
    try:
        test_record = blockchain.store_attendance("INTEGRATION_TEST", True, 0.99)
        print(f"✅ Blockchain storage working")
        print(f"   Hash: {test_record['blockchain_hash'][:20]}...")
        print(f"   TX: {test_record['transaction_hash'][:20]}...")
    except Exception as e:
        print(f"❌ Blockchain storage failed: {e}")
        return False
    
    print("-" * 50)
    print("🎉 BLOCKCHAIN FULLY INTEGRATED!")
    return True

if __name__ == "__main__":
    check_blockchain_integration()
