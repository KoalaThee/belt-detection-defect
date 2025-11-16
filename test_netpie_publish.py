#!/usr/bin/env python3
"""
Test script to manually test NetPIE data publishing.
Run this while Flask app is running to test if data publishing works.
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("✓ Loaded .env file")
    else:
        print("✗ .env file not found")
        sys.exit(1)
except ImportError:
    print("✗ python-dotenv not installed")
    sys.exit(1)

# Import NetPIE client
try:
    import netpie_client
    print("✓ NetPIE client imported")
except ImportError as e:
    print(f"✗ Failed to import netpie_client: {e}")
    sys.exit(1)

# Check connection status
print("\n" + "=" * 60)
print("NetPIE Publishing Test")
print("=" * 60)

if not netpie_client.is_connected():
    print("\n⚠️  NetPIE is not connected!")
    print("Attempting to connect...")
    
    import os
    app_id = os.environ.get('NETPIE_APP_ID')
    app_key = os.environ.get('NETPIE_APP_KEY')
    app_secret = os.environ.get('NETPIE_APP_SECRET')
    
    if not all([app_id, app_key, app_secret]):
        print("✗ Missing NetPIE credentials in .env file")
        sys.exit(1)
    
    if netpie_client.initialize_netpie(app_id, app_key, app_secret):
        print("✓ Connected to NetPIE successfully")
    else:
        print("✗ Failed to connect to NetPIE")
        sys.exit(1)
else:
    print("\n✓ NetPIE is already connected")

# Test 1: Regular publish function
print("\n" + "-" * 60)
print("Test 1: Testing regular publish_defect_data()")
print("-" * 60)
result1 = netpie_client.test_publish()
print(f"Result: {'✓ Success' if result1 else '✗ Failed'}")

# Test 2: Force publish to @shadow/data/update
print("\n" + "-" * 60)
print("Test 2: Force publishing to @shadow/data/update")
print("-" * 60)
result2 = netpie_client.force_publish_test()
print(f"Result: {'✓ Success' if result2 else '✗ Failed'}")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print(f"Regular publish: {'✓ Pass' if result1 else '✗ Fail'}")
print(f"Force publish:   {'✓ Pass' if result2 else '✗ Fail'}")

if result1 or result2:
    print("\n✓ At least one publish method succeeded!")
    print("Check your NetPIE portal shadow data in a few seconds.")
    print("Look for data under '@shadow/data/update' or '@msg/defect_detection'")
else:
    print("\n✗ Both publish methods failed.")
    print("Check the logs above for error details.")

print("\n" + "=" * 60)

