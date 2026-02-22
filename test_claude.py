#!/usr/bin/env python
"""
Temporary test script using Claude instead of DeepSeek R1
This allows you to test the YouTube summarization functionality while waiting for DeepSeek access.
"""
import os
import sys
import json
from pathlib import Path

# Add the current directory to Python path to import from youtubeSummarize
sys.path.insert(0, str(Path(__file__).parent))

try:
    from youtubeSummarize import create_bedrock_client
    print("✓ Successfully imported Bedrock functions")
except ImportError as e:
    print(f"✗ Failed to import Bedrock functions: {e}")
    sys.exit(1)

def invoke_claude_model(client, messages: list, max_tokens: int = 2000, temperature: float = 0.3) -> str:
    """Invoke Claude model on AWS Bedrock."""
    try:
        # Format messages for Claude API
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1",
            body=json.dumps(body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text'].strip()
        
    except Exception as e:
        print(f"Error invoking Claude model: {e}")
        raise

def test_claude_connection():
    """Test Claude model connection and invocation."""
    try:
        print("Testing AWS Bedrock connection with Claude...")
        client = create_bedrock_client()
        print("✓ Bedrock client created successfully")
        
        # Test with a simple message
        test_messages = [
            {"role": "user", "content": "Say 'Hello, Claude!' and nothing else."}
        ]
        
        print("Testing Claude model invocation...")
        response = invoke_claude_model(client, test_messages, max_tokens=50, temperature=0.1)
        print(f"✓ Claude response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure AWS credentials are configured")
        print("2. Ensure you have access to Claude model in Bedrock")
        print("3. Check that your AWS region supports Claude")
        print("4. Verify your AWS account has Bedrock permissions")
        return False

if __name__ == "__main__":
    print("AWS Bedrock Claude Integration Test")
    print("=" * 50)
    
    # Check AWS credentials
    if not os.environ.get('AWS_ACCESS_KEY_ID') and not os.path.exists(os.path.expanduser('~/.aws/credentials')):
        print("⚠️  Warning: No AWS credentials found in environment or ~/.aws/credentials")
        print("   Please configure AWS credentials before running this test")
        sys.exit(1)
    
    success = test_claude_connection()
    
    if success:
        print("\n🎉 Claude test passed! You can use Claude as a temporary alternative.")
        print("To use Claude instead of DeepSeek R1, update the model ID in youtubeSummarize.py:")
        print("Change: DEEPSEEK_MODEL_ID = \"us.deepseek.r1-v1:0\"")
        print("To:     DEEPSEEK_MODEL_ID = \"anthropic.claude-3-sonnet-20240229-v1\"")
    else:
        print("\n❌ Claude test also failed. Please check AWS Bedrock permissions.")
        sys.exit(1)
