pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        DOCKER_CREDENTIALS = credentials('docker-registry-credentials')
        KUBE_CONFIG = credentials('kubernetes-config')
        SLACK_CHANNEL = '#fraud-prevention-deployments'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: "git rev-parse --short HEAD",
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'npm ci'
            }
        }
        
        stage('Lint') {
            steps {
                sh 'npm run lint'
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm run test:unit'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'npm run test:integration'
                    }
                }
            }
        }
        
        stage('Build Services') {
            parallel {
                stage('Biometric Service') {
                    steps {
                        sh '''
                            docker build -f infrastructure/docker/biometric-service.Dockerfile \
                                -t ${DOCKER_REGISTRY}/biometric-service:${GIT_COMMIT_SHORT} \
                                -t ${DOCKER_REGISTRY}/biometric-service:latest .
                        '''
                    }
                }
                stage('Fraud Detection Service') {
                    steps {
                        sh '''
                            docker build -f infrastructure/docker/fraud-detection.Dockerfile \
                                -t ${DOCKER_REGISTRY}/fraud-detection-service:${GIT_COMMIT_SHORT} \
                                -t ${DOCKER_REGISTRY}/fraud-detection-service:latest .
                        '''
                    }
                }
            }
        }
        
        stage('Push Images') {
            steps {
                sh '''
                    echo ${DOCKER_CREDENTIALS_PSW} | docker login ${DOCKER_REGISTRY} -u ${DOCKER_CREDENTIALS_USR} --password-stdin
                    docker push ${DOCKER_REGISTRY}/biometric-service:${GIT_COMMIT_SHORT}
                    docker push ${DOCKER_REGISTRY}/biometric-service:latest
                    docker push ${DOCKER_REGISTRY}/fraud-detection-service:${GIT_COMMIT_SHORT}
                    docker push ${DOCKER_REGISTRY}/fraud-detection-service:latest
                '''
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh '''
                    kubectl --kubeconfig=${KUBE_CONFIG} set image deployment/biometric-service \
                        biometric-service=${DOCKER_REGISTRY}/biometric-service:${GIT_COMMIT_SHORT} \
                        -n fraud-prevention-staging
                    kubectl --kubeconfig=${KUBE_CONFIG} set image deployment/fraud-detection-service \
                        fraud-detection-service=${DOCKER_REGISTRY}/fraud-detection-service:${GIT_COMMIT_SHORT} \
                        -n fraud-prevention-staging
                '''
            }
        }
        
        stage('Smoke Tests') {
            steps {
                sh 'npm run test:smoke'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                sh '''
                    kubectl --kubeconfig=${KUBE_CONFIG} set image deployment/biometric-service \
                        biometric-service=${DOCKER_REGISTRY}/biometric-service:${GIT_COMMIT_SHORT} \
                        -n fraud-prevention
                    kubectl --kubeconfig=${KUBE_CONFIG} set image deployment/fraud-detection-service \
                        fraud-detection-service=${DOCKER_REGISTRY}/fraud-detection-service:${GIT_COMMIT_SHORT} \
                        -n fraud-prevention
                '''
            }
        }
    }
    
    post {
        success {
            slackSend(
                channel: env.SLACK_CHANNEL,
                color: 'good',
                message: "Deployment successful: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
            )
        }
        failure {
            slackSend(
                channel: env.SLACK_CHANNEL,
                color: 'danger',
                message: "Deployment failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
            )
        }
    }
}
