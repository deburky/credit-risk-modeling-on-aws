# Credit Risk Modeling on AWS

This repository contains code examples and implementations from the **Credit Risk Modeling on AWS** book project.

## Overview

This project demonstrates real-time credit risk scoring using AWS services, including model training, deployment, and inference pipelines.

## Project Structure

```
.
├── data/                    # Sample credit data
│   └── credit_example.csv
└── real_time_scoring/       # Real-time scoring implementation
    ├── README.md           # Detailed documentation
    ├── Makefile            # Build and deployment commands
    └── ...
```

## Getting Started

### Real-Time Credit Scoring

The `real_time_scoring/` folder contains a complete implementation of a real-time credit scoring system using:

- **LocalStack** for local AWS service emulation
- **Kinesis** for streaming loan applications
- **Lambda** for real-time scoring
- **DynamoDB** for storing approved applications
- **CloudFormation** for infrastructure as code

See the [real_time_scoring/README.md](real_time_scoring/README.md) for detailed instructions.

## Data

The `data/` directory contains sample credit application data used for training and testing the models.

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

