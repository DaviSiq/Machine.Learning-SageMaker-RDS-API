import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# Definir variáveis de ambiente para as credenciais e região
import boto3
import sagemaker
from sagemaker import Session
s3 = boto3.client('s3')
response = s3.list_buckets()
for bucket in response['Buckets'] :
    print(f"{bucket['Name']}")

boto_3 = boto3.Session()
session = sagemaker.Session(boto_3)

# Configurar o SageMaker
sess = sagemaker.Session()
role = 'arn:aws:iam::992382374294:role/service-role/AmazonSageMaker-ExecutionRole-20240615T181656'
bucket = 'pb-aws-teste2-s3'
train_data = f's3://{bucket}/modified_hotel_reservations.csv'

# Configurar o estimador SKLearn
FRAMEWORK_VERSION = "0.23-1"
sklearn_estimator = SKLearn(
    entry_point="treinamento_script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="RF-custom-sklearn",
    hyperparameters={
        "n_estimators": 100,
        "random_state": 0,
    },
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600,
    sagemaker_session=sess
)

# Configurar input do treinamento
train_input = TrainingInput(train_data, content_type='csv')

# Iniciar o treinamento
sklearn_estimator.fit({'train': train_input})
