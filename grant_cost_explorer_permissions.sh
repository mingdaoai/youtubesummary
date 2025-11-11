#!/bin/bash
# Script to grant Cost Explorer permissions to AWS IAM user
# This must be run by an AWS administrator

USER_NAME="ken"
POLICY_NAME="CostExplorerReadOnly"

echo "Granting Cost Explorer permissions to user: $USER_NAME"
echo "=================================================================="

# Create policy document
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

echo ""
echo "Policy document created at: /tmp/cost_explorer_policy.json"
echo ""
echo "To grant permissions, run one of these commands as an AWS administrator:"
echo ""
echo "Option 1: Add as inline policy (recommended for single user):"
echo "  aws iam put-user-policy --user-name $USER_NAME --policy-name $POLICY_NAME --policy-document file:///tmp/cost_explorer_policy.json"
echo ""
echo "Option 2: Create managed policy and attach (recommended for multiple users):"
echo "  aws iam create-policy --policy-name $POLICY_NAME --policy-document file:///tmp/cost_explorer_policy.json"
echo "  aws iam attach-user-policy --user-name $USER_NAME --policy-arn arn:aws:iam::ACCOUNT_ID:policy/$POLICY_NAME"
echo ""
echo "Option 3: Via AWS Console:"
echo "  1. Go to: https://console.aws.amazon.com/iam/"
echo "  2. Click 'Users' -> Select user '$USER_NAME'"
echo "  3. Click 'Add permissions' -> 'Create inline policy'"
echo "  4. Switch to JSON tab"
echo "  5. Paste the contents of /tmp/cost_explorer_policy.json"
echo "  6. Name it: $POLICY_NAME"
echo "  7. Click 'Create policy'"
echo ""

# Try to add it if we have permissions
if aws iam put-user-policy --user-name "$USER_NAME" --policy-name "$POLICY_NAME" --policy-document file:///tmp/cost_explorer_policy.json 2>/dev/null; then
    echo "✅ Successfully added Cost Explorer permissions!"
    echo ""
    echo "Testing access..."
    if aws ce get-cost-and-usage --time-period Start=2025-10-11,End=2025-11-10 --granularity MONTHLY --metrics UnblendedCost --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Transcribe"]}}' 2>/dev/null; then
        echo "✅ Cost Explorer access confirmed!"
    else
        echo "⚠️  Permissions added but Cost Explorer may need to be enabled in console"
    fi
else
    echo "⚠️  Could not add permissions automatically (requires admin access)"
    echo "   Please use one of the options above"
fi

