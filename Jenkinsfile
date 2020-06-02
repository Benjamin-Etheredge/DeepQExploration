pipeline {
  agent {
    dockerfile {
      filename 'Dockerfile'
    }

  }
  stages {
    stage('Test') {
      steps {
        echo 'Step 2'
        sh 'python -m unittest test_agent.TestAgent.test_vanilla'
      }
    }

  }
}