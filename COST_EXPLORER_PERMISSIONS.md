# Granting Cost Explorer Permissions

The AWS user `ken` needs Cost Explorer permissions to query billing data.

## Quick Method (AWS Console)

1. Go to: https://console.aws.amazon.com/iam/
2. Click **Users** in the left menu
3. Select user: **ken**
4. Click **Add permissions** → **Create inline policy**
5. Switch to the **JSON** tab
6. Paste this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ce:GetCostAndUsage",
        "ce:GetUsageReport",
        "ce:GetDimensionValues",
        "ce:GetReservationCoverage",
        "ce:GetReservationPurchaseRecommendation",
        "ce:GetReservationUtilization",
        "ce:GetRightsizingRecommendation",
        "ce:GetSavingsPlansCoverage",
        "ce:GetSavingsPlansUtilization",
        "ce:GetSavingsPlansUtilizationDetails",
        "ce:ListCostCategoryDefinitions"
      ],
      "Resource": "*"
    }
  ]
}
```

7. Click **Next**
8. Name the policy: `CostExplorerReadOnly`
9. Click **Create policy**

## Command Line Method (Requires Admin Access)

```bash
# Create policy file
cat > /tmp/cost_explorer_policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ce:GetCostAndUsage",
        "ce:GetUsageReport",
        "ce:GetDimensionValues",
        "ce:GetReservationCoverage",
        "ce:GetReservationPurchaseRecommendation",
        "ce:GetReservationUtilization",
        "ce:GetRightsizingRecommendation",
        "ce:GetSavingsPlansCoverage",
        "ce:GetSavingsPlansUtilization",
        "ce:GetSavingsPlansUtilizationDetails",
        "ce:ListCostCategoryDefinitions"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Add as inline policy
aws iam put-user-policy \
  --user-name ken \
  --policy-name CostExplorerReadOnly \
  --policy-document file:///tmp/cost_explorer_policy.json
```

## Verify Permissions

After adding permissions, test with:

```bash
python3 check_aws_transcribe_cost.py
```

Or directly:

```bash
aws ce get-cost-and-usage \
  --time-period Start=2025-10-11,End=2025-11-10 \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Transcribe"]}}'
```

## Enable Cost Explorer (If Not Already Enabled)

Cost Explorer may need to be enabled in the AWS Console:

1. Go to: https://console.aws.amazon.com/cost-management/home
2. Click **Cost Explorer** in the left menu
3. If you see "Enable Cost Explorer", click it
4. Wait 24 hours for data to populate

## Notes

- The policy grants read-only access to Cost Explorer APIs
- No write/modify permissions are included
- Safe to add to any user who needs to view billing data

