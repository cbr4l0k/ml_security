"""CLI entry point for the ML Security Challenge Suite."""

import argparse
import sys

from ml_security.client import ChallengeClient


def main():
    parser = argparse.ArgumentParser(
        description="ML Security Challenge Suite — solve adversarial ML challenges",
    )
    parser.add_argument("--challenge", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Challenge number to solve (1-4)")
    parser.add_argument("--api-key", required=True, help="API key for the platform")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru", help="Base API URL")
    parser.add_argument("--username", help="Username for JWT authentication")
    parser.add_argument("--password", help="Password for JWT authentication")

    # Challenge-specific arguments
    parser.add_argument("--image", help="[Challenge 1] Path to source image")
    parser.add_argument("--image-id", default="challenge_001", help="[Challenge 1] Image identifier")
    parser.add_argument("--output-dir", default="output", help="[Challenge 2] Output directory for images/masks")
    parser.add_argument("--num-queries", type=int, default=500, help="[Challenge 3] Number of queries to make")
    parser.add_argument("--model-path", default="backdoored_model.pt", help="[Challenge 4] Path to save model")

    args = parser.parse_args()

    client = ChallengeClient(
        api_url=args.api_url,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
    )

    if args.challenge == 1:
        if not args.image:
            print("Error: --image is required for challenge 1", file=sys.stderr)
            sys.exit(1)
        from solvers.solve_challenge_1 import solve
        solve(client, args.image, args.image_id)

    elif args.challenge == 2:
        from solvers.solve_challenge_2 import solve
        solve(client, args.output_dir)

    elif args.challenge == 3:
        from solvers.solve_challenge_3 import solve
        solve(client, args.num_queries)

    elif args.challenge == 4:
        from solvers.solve_challenge_4 import solve
        solve(client, args.model_path)


if __name__ == "__main__":
    main()
