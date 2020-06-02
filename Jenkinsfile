pipeline {
  agent {
    dockerfile {
      filename 'Dockerfile'
    }

  }
  stages {
    stage('Build') {
      steps {
        sh 'make build'
        echo 'Made container'
      }
    }

    stage('Test') {
      steps {
        echo 'Step 2'
      }
    }

  }
}