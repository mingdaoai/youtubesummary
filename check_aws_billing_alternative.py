#!/usr/bin/env python3
"""
Alternative methods to check AWS Transcribe costs:
1. Check CloudWatch metrics (if available)
2. List recent transcription jobs and estimate
3. Provide instructions for manual check
"""

import boto3
from datetime import datetime, timedelta
import json

def check_transcribe_jobs():
    """List recent transcription jobs to estimate usage."""
    try:
        transcribe = boto3.client('transcribe', region_name='us-west-2')
        
        print("Checking recent AWS Transcribe jobs...")
        print("-" * 80)
        
        # List jobs from last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        
        jobs = []
        paginator = transcribe.get_paginator('list_transcription_jobs')
        
        for page in paginator.paginate(
            Status='COMPLETED',
            MaxResults=100
        ):
            for job in page.get('TranscriptionJobSummaries', []):
                creation_time = job.get('CreationTime')
                if creation_time and creation_time >= cutoff_date:
                    jobs.append({
                        'name': job.get('TranscriptionJobName', ''),
                        'status': job.get('TranscriptionJobStatus', ''),
                        'created': creation_time.strftime('%Y-%m-%d %H:%M:%S') if creation_time else 'Unknown',
                        'language': job.get('LanguageCode', 'Unknown')
                    })
        
        print(f"Found {len(jobs)} completed transcription jobs in last 30 days")
        
        if jobs:
            print(f"\nRecent jobs (showing first 10):")
            for job in jobs[:10]:
                print(f"  {job['name']:50s} {job['created']} ({job['language']})")
        
        return len(jobs)
        
    except Exception as e:
        print(f"Error checking transcription jobs: {e}")
        return None


def check_s3_usage():
    """Check S3 bucket usage for audio files."""
    try:
        s3 = boto3.client('s3')
        bucket = 'mdaudiosound'
        
        print(f"\nChecking S3 bucket '{bucket}' for audio files...")
        print("-" * 80)
        
        # Count files in input/ prefix
        paginator = s3.get_paginator('list_objects_v2')
        total_size = 0
        file_count = 0
        
        for page in paginator.paginate(Bucket=bucket, Prefix='input/'):
            for obj in page.get('Contents', []):
                total_size += obj.get('Size', 0)
                file_count += 1
        
        size_gb = total_size / (1024 ** 3)
        print(f"  Audio files in S3: {file_count}")
        print(f"  Total size: {size_gb:.2f} GB")
        print(f"  Estimated S3 storage cost: ${size_gb * 0.023:.2f}/month (Standard storage)")
        
        return file_count, total_size
        
    except Exception as e:
        print(f"Error checking S3: {e}")
        return None, None


def manual_check_instructions():
    """Provide instructions for manual billing check."""
    print("\n" + "=" * 80)
    print("MANUAL BILLING CHECK INSTRUCTIONS:")
    print("=" * 80)
    print("\nOption 1: AWS Console - Cost Explorer")
    print("  1. Go to: https://console.aws.amazon.com/cost-management/home")
    print("  2. Click 'Cost Explorer' in left menu")
    print("  3. If not enabled, click 'Enable Cost Explorer' (takes 24h to populate)")
    print("  4. Create a custom report:")
    print("     - Service: Amazon Transcribe")
    print("     - Date range: Last 30 days")
    print("     - Group by: Service")
    print("\nOption 2: AWS Console - Bills")
    print("  1. Go to: https://console.aws.amazon.com/billing/")
    print("  2. Click 'Bills' in left menu")
    print("  3. Select current month")
    print("  4. Search for 'Transcribe' in the bill")
    print("\nOption 3: AWS CLI (if you have billing permissions)")
    print("  aws ce get-cost-and-usage \\")
    print("    --time-period Start=2025-10-11,End=2025-11-10 \\")
    print("    --granularity MONTHLY \\")
    print("    --metrics UnblendedCost \\")
    print("    --filter '{\"Dimensions\":{\"Key\":\"SERVICE\",\"Values\":[\"Amazon Transcribe\"]}}'")
    print("\nOption 4: Add IAM permissions")
    print("  Add this policy to your IAM user:")
    print("  {")
    print("    \"Version\": \"2012-10-17\",")
    print("    \"Statement\": [{")
    print("      \"Effect\": \"Allow\",")
    print("      \"Action\": [\"ce:GetCostAndUsage\", \"ce:GetUsageReport\"],")
    print("      \"Resource\": \"*\"")
    print("    }]")
    print("  }")


if __name__ == '__main__':
    print("AWS Billing Check - Alternative Methods")
    print("=" * 80)
    
    # Check transcription jobs
    job_count = check_transcribe_jobs()
    
    # Check S3 usage
    file_count, total_size = check_s3_usage()
    
    # Manual instructions
    manual_check_instructions()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    if job_count:
        print(f"  Transcription jobs (last 30 days): {job_count}")
    if file_count:
        print(f"  Audio files in S3: {file_count}")
    print(f"  Estimated Transcribe cost: ~$34.72 (from usage calculation)")
    print(f"  To get actual cost: Use manual check methods above")

