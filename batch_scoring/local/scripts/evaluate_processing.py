"""
Processing script to evaluate limit increase eligibility
"""

import json
import os


def main():
    """Main processing function"""
    # Load scored customers from input (handle S3 download structure)
    input_dir = "/opt/ml/processing/input/scored_customers"

    # Find the JSON file (could be in subdirectories from S3)
    scored_file = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                scored_file = os.path.join(root, file)
                break
        if scored_file:
            break

    if not scored_file:
        # Try direct path
        scored_file = os.path.join(input_dir, "scored_customers.json")

    print(f"Loading scored customers from {scored_file}...")
    with open(scored_file, "r") as f:
        scored_customers = json.load(f)

    print(f"Found {len(scored_customers)} customers to evaluate")

    # Evaluate limit increases
    evaluations = []

    for customer in scored_customers:
        customer_id = customer["customer_id"]
        current_score = customer["current_score"]
        new_score = customer["new_score"]
        current_limit = customer["current_limit"]

        # Business rules
        if new_score >= 650:
            # Eligible for increase
            if current_limit < 5000:
                increase_percent = 0.50  # 50% increase
            else:
                increase_percent = 0.25  # 25% increase

            new_limit = int(current_limit * (1 + increase_percent))
            new_limit = min(new_limit, 20000)  # Cap at 20000

            decision = "APPROVED"
            reason = f"Score {new_score:.0f} >= 650, increase by {increase_percent * 100:.0f}%"
        else:
            # Not eligible
            new_limit = current_limit
            decision = "DECLINED"
            reason = f"Score {new_score:.0f} < 650, no increase"

        evaluations.append(
            {
                "customer_id": customer_id,
                "current_score": current_score,
                "new_score": new_score,
                "current_limit": current_limit,
                "new_limit": new_limit,
                "decision": decision,
                "reason": reason,
            }
        )

    approved_count = sum(1 for e in evaluations if e["decision"] == "APPROVED")
    declined_count = len(evaluations) - approved_count

    print(
        f"✓ Evaluated {len(evaluations)} customers: {approved_count} approved, {declined_count} declined"
    )

    # Save to output
    output_path = "/opt/ml/processing/output/evaluations.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(evaluations, f, indent=2)

    print(f"✓ Evaluations saved to {output_path}")


if __name__ == "__main__":
    main()
