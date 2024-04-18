# enablement-session-01
enablement-session-01
- Reference updated GenAIHub: https://github.wdf.sap.corp/AI/generative-ai-hub-sdk/blob/main/docs/gen_ai_hub/examples/gen_ai_hub.ipynb
- Dataset: IKEA Products dataset + Block dataset powered by GPT

## Vector demo1 (Dataset; DEMO_BLOCK_1: Old block issue dataset)
- SQL based similiarty search on vector embeddings in HANA 
- Embedding.txt & Console.sql 

## Vector demo2 (Dataset; DEMO_BLOCK_3: New Block reason dataset for enablement session)

## BTP Kyma Runtime 
- config name: kubeconfig--garden-kyma--c-5d3089e-external.yaml
- config file path: ./kube/kubeconfig--garden-kyma--c-5d3089e-external.yaml
- Steps
1. export KUBECONFIG=./kube/kubeconfig.yaml    #Kube configuration 
2. kubectl config get-contexts 
3. kubectl version
4. docker-credential-osxkeychain list
5. kubectl create -f k8s/deployment.yaml
6. kubectl apply -f k8s/simulator-deploy.yaml
