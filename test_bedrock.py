#!/usr/bin/env python
"""
Test script to verify AWS Bedrock integration with Claude 3.5 Sonnet
"""
import os
import sys
import json
from pathlib import Path

# Add the current directory to Python path to import from youtubeSummarize
sys.path.insert(0, str(Path(__file__).parent))

try:
    from youtubeSummarize import create_bedrock_client, invoke_bedrock_model
    print("✓ Successfully imported Bedrock functions")
except ImportError as e:
    print(f"✗ Failed to import Bedrock functions: {e}")
    sys.exit(1)

def test_bedrock_connection():
    """Test basic Bedrock connection and model invocation."""
    try:
        print("Testing AWS Bedrock connection...")
        client = create_bedrock_client()
        print("✓ Bedrock client created successfully")
        
        # Test with a simple message
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, Bedrock!' and nothing else."}
        ]
        
        print("Testing model invocation...")
        response = invoke_bedrock_model(client, test_messages, max_tokens=50, temperature=0.1)
        print(f"✓ Model response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure AWS credentials are configured:")
        print("   - Run 'aws configure' or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("2. Ensure you have access to Claude 3.5 Sonnet model in Bedrock")
        print("3. Check that your AWS region supports Claude 3.5 Sonnet (us-west-2, us-east-1, us-east-2, eu-west-1, ap-northeast-1)")
        print("4. Verify your AWS account has Bedrock permissions")
        return False

if __name__ == "__main__":
    print("AWS Bedrock Claude 3.5 Sonnet Integration Test")
    print("=" * 50)
    
    # Check AWS credentials
    if not os.environ.get('AWS_ACCESS_KEY_ID') and not os.path.exists(os.path.expanduser('~/.aws/credentials')):
        print("⚠️  Warning: No AWS credentials found in environment or ~/.aws/credentials")
        print("   Please configure AWS credentials before running this test")
        sys.exit(1)
    
    success = test_bedrock_connection()
    
    if success:
        print("\n🎉 All tests passed! Claude 3.5 Sonnet on AWS Bedrock is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)
