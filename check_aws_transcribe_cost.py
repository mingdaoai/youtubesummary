#!/usr/bin/env python3
"""
Query AWS Cost Explorer to get actual AWS Transcribe costs.
"""

import boto3
from datetime import datetime, timedelta
import json

def get_transcribe_costs(start_date=None, end_date=None):
    """
    Query AWS Cost Explorer for AWS Transcribe costs.
    
    Args:
        start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
        end_date: End date (YYYY-MM-DD), defaults to today
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        ce = boto3.client('ce', region_name='us-east-1')  # Cost Explorer is global, us-east-1 is standard
        
        print(f"Querying AWS Cost Explorer for Transcribe costs...")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 80)
        
        # Query for AWS Transcribe service costs
        response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon Transcribe']
                }
            }
        )
        
        # Parse results
        total_cost = 0.0
        daily_costs = []
        
        for result in response.get('ResultsByTime', []):
            date = result['TimePeriod']['Start']
            cost = float(result['Total']['UnblendedCost']['Amount'])
            total_cost += cost
            if cost > 0:
                daily_costs.append({
                    'date': date,
                    'cost': cost
                })
        
        print(f"\nAWS Transcribe Costs:")
        print(f"  Total cost: ${total_cost:.2f}")
        print(f"  Date range: {start_date} to {end_date}")
        
        if daily_costs:
            print(f"\nDaily breakdown (non-zero days):")
            for day in sorted(daily_costs, key=lambda x: x['date']):
                print(f"  {day['date']}: ${day['cost']:.4f}")
        else:
            print(f"\nNo costs found for this period.")
        
        # Also try to get usage statistics
        print(f"\n" + "-" * 80)
        print(f"Attempting to get usage statistics...")
        
        try:
            # Get usage for Transcribe
            usage_response = ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UsageQuantity'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Transcribe']
                    }
                }
            )
            
            total_usage = 0.0
            for result in usage_response.get('ResultsByTime', []):
                usage = float(result['Total']['UsageQuantity']['Amount'])
                total_usage += usage
            
            if total_usage > 0:
                print(f"  Total usage (minutes): {total_usage:.1f}")
                print(f"  Average cost per minute: ${total_cost/total_usage:.4f}" if total_usage > 0 else "")
        except Exception as e:
            print(f"  Could not get usage statistics: {e}")
        
        return {
            'total_cost': total_cost,
            'start_date': start_date,
            'end_date': end_date,
            'daily_costs': daily_costs
        }
        
    except Exception as e:
        print(f"Error querying AWS Cost Explorer: {e}")
        print(f"\nPossible reasons:")
        print(f"  1. AWS credentials not configured")
        print(f"  2. Cost Explorer API not enabled (first time use requires activation)")
        print(f"  3. Insufficient permissions (need ce:GetCostAndUsage)")
        print(f"\nTo enable Cost Explorer:")
        print(f"  1. Go to AWS Console -> Cost Management -> Cost Explorer")
        print(f"  2. Click 'Enable Cost Explorer'")
        print(f"  3. Wait 24 hours for data to populate")
        return None


def get_all_aws_costs(start_date=None, end_date=None):
    """
    Get all AWS costs (not just Transcribe) for comparison.
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        ce = boto3.client('ce', region_name='us-east-1')
        
        print(f"\n" + "=" * 80)
        print(f"All AWS Costs (for comparison):")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 80)
        
        response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        services = []
        for result in response.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                if cost > 0:
                    services.append({
                        'service': service,
                        'cost': cost
                    })
        
        services.sort(key=lambda x: x['cost'], reverse=True)
        
        total_all = sum(s['cost'] for s in services)
        print(f"\nTop services by cost:")
        for service in services[:10]:
            percentage = (service['cost'] / total_all * 100) if total_all > 0 else 0
            print(f"  {service['service']:40s} ${service['cost']:8.2f} ({percentage:5.1f}%)")
        
        print(f"\n  {'Total AWS costs':40s} ${total_all:8.2f}")
        
        return services
        
    except Exception as e:
        print(f"Error querying all AWS costs: {e}")
        return None


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Query AWS Cost Explorer for Transcribe costs')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD), defaults to 30 days ago')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--all', action='store_true', help='Also show all AWS costs')
    
    args = parser.parse_args()
    
    # Get Transcribe costs
    transcribe_costs = get_transcribe_costs(args.start_date, args.end_date)
    
    # Get all costs if requested
    if args.all:
        all_costs = get_all_aws_costs(args.start_date, args.end_date)
    
    if transcribe_costs:
        print(f"\n" + "=" * 80)
        print(f"Summary:")
        print(f"  Estimated from usage: ~$34.72")
        print(f"  Actual from AWS billing: ${transcribe_costs['total_cost']:.2f}")
        if transcribe_costs['total_cost'] > 0:
            diff = abs(34.72 - transcribe_costs['total_cost'])
            print(f"  Difference: ${diff:.2f}")
            if diff < 5:
                print(f"  ✅ Estimates are close to actual costs!")
            else:
                print(f"  ⚠️  Significant difference - may need to check other factors")

