"""
Train credit scorecard using SageMaker Local Mode
Stores model in LocalStack S3
"""

import logging
import os
from sagemaker.sklearn import SKLearn
from sagemaker.local import LocalSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_sagemaker_local():
    """Train scorecard using SageMaker Local Mode"""
    
    logger.info("=" * 80)
    logger.info("Training Credit Scorecard with SageMaker Local Mode")
    logger.info("=" * 80)
    
    # Use LocalSession for local training
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}
    
    # Define paths
    train_data = 'file://../data'  # Local path to training data
    output_path = 'file://./model_output'  # Local output
    
    # Create SKLearn estimator
    sklearn_estimator = SKLearn(
        entry_point='training/train.py',
        role='arn:aws:iam::000000000000:role/SageMakerRole',  # Dummy role for local
        instance_type='local',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        hyperparameters={
            'target-score': 600,
            'target-odds': 30,
            'pts-double-odds': 20,
        }
    )
    
    logger.info("\nStarting SageMaker Local training...")
    logger.info(f"Training data: {train_data}")
    logger.info(f"Output path: {output_path}")
    
    # Train
    sklearn_estimator.fit({'train': train_data})
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info(f"Model artifacts: {output_path}")
    logger.info("=" * 80)
    
    return sklearn_estimator


if __name__ == '__main__':
    estimator = train_with_sagemaker_local()




