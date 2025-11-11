#!/usr/bin/env python
"""
AWS Bedrock Setup Helper Script
This script helps diagnose and fix AWS Bedrock setup issues.
"""
import boto3
import json
import sys

def check_aws_credentials():
    """Check if AWS credentials are properly configured."""
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            print("✓ AWS credentials found")
            return True
        else:
            print("✗ No AWS credentials found")
            return False
    except Exception as e:
        print(f"✗ Error checking credentials: {e}")
        return False

def check_bedrock_permissions():
    """Check what Bedrock permissions the current user has."""
    try:
        client = boto3.client('bedrock', region_name='us-east-1')
        
        # Try to list foundation models
        try:
            models = client.list_foundation_models()
            print("✓ Has bedrock:ListFoundationModels permission")
            return True
        except Exception as e:
            if "AccessDeniedException" in str(e):
                print("✗ Missing bedrock:ListFoundationModels permission")
            else:
                print(f"✗ Error listing models: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error creating Bedrock client: {e}")
        return False

def check_model_access():
    """Check if we can access the DeepSeek model."""
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Try a simple test invocation
        test_body = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = client.invoke_model(
            modelId="us.deepseek.r1-v1:0",
            body=json.dumps(test_body),
            contentType="application/json"
        )
        
        print("✓ DeepSeek R1 model access confirmed")
        return True
        
    except Exception as e:
        if "AccessDeniedException" in str(e):
            print("✗ Missing bedrock:InvokeModel permission or model access not granted")
        elif "ValidationException" in str(e):
            print("✗ Model ID invalid or model not available")
        else:
            print(f"✗ Error testing model access: {e}")
        return False

def print_setup_instructions():
    """Print detailed setup instructions."""
    print("\n" + "="*60)
    print("AWS BEDROCK SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. REQUEST MODEL ACCESS:")
    print("   - Go to: https://console.aws.amazon.com/bedrock/")
    print("   - Navigate to 'Model access' under 'Bedrock configurations'")
    print("   - Find 'DeepSeek' provider and request access to 'DeepSeek R1'")
    print("   - Wait for AWS approval (usually minutes to hours)")
    
    print("\n2. ADD IAM PERMISSIONS:")
    print("   Create or update your IAM policy with these permissions:")
    print("   {")
    print('     "Version": "2012-10-17",')
    print('     "Statement": [')
    print('       {')
    print('         "Effect": "Allow",')
    print('         "Action": [')
    print('           "bedrock:InvokeModel",')
    print('           "bedrock:ListFoundationModels"')
    print('         ],')
    print('         "Resource": "*"')
    print('       }')
    print('     ]')
    print("   }")
    
    print("\n3. ALTERNATIVE MODELS:")
    print("   If you want to test immediately, you can use other models:")
    print("   - Claude: anthropic.claude-3-sonnet-20240229-v1:0")
    print("   - Llama: meta.llama2-13b-chat-v1")
    print("   - Titan: amazon.titan-text-express-v1")
    
    print("\n4. REGIONS:")
    print("   DeepSeek R1 is available in:")
    print("   - us-east-1 (N. Virginia)")
    print("   - us-east-2 (Ohio)")
    print("   - us-west-2 (Oregon)")
    
    print("\n5. TEST COMMANDS:")
    print("   After setup, test with:")
    print("   python test_bedrock.py")

def main():
    print("AWS Bedrock Setup Diagnostic Tool")
    print("="*40)
    
    # Check credentials
    creds_ok = check_aws_credentials()
    
    # Check permissions
    perms_ok = check_bedrock_permissions()
    
    # Check model access
    model_ok = check_model_access()
    
    # Print results
    print(f"\nSUMMARY:")
    print(f"Credentials: {'✓' if creds_ok else '✗'}")
    print(f"Permissions: {'✓' if perms_ok else '✗'}")
    print(f"Model Access: {'✓' if model_ok else '✗'}")
    
    if not (creds_ok and perms_ok and model_ok):
        print_setup_instructions()
        return False
    else:
        print("\n🎉 All checks passed! Your Bedrock setup is ready.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
