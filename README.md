# Week 6 — Continuous Deployment: IRIS Inference API on GKE

In this assignment, I integrated Continuous Deployment (CD) into my IRIS ML pipeline by containerizing the inference API with Docker and deploying it to Google Kubernetes Engine (GKE) — with the entire build, push, and deploy cycle automated through GitHub Actions.

---

## What I Did

Following the CI pipeline and MLflow integration from previous weeks, I completed the final loop: getting a validated model into a live, scalable prediction service automatically on every code push.

---

## Tasks Completed

### Task 1 — Pod vs Container (Video Explanation)
I explained the difference between a Kubernetes Pod and a Docker container in my screencast:
- A **Docker container** is a running instance of an image — isolated, lightweight, and single-process focused.
- A **Kubernetes Pod** is the smallest deployable unit in K8s — it wraps one or more containers that share the same network namespace and storage volumes.
- Kubernetes uses Pods (not bare containers) to allow tightly coupled processes to communicate over `localhost`, and to give the scheduler a consistent unit to place, scale, and manage.

### Task 2 — Dockerfile for IRIS API
I wrote a `Dockerfile` that:
- Uses a Python base image
- Installs all inference dependencies from `requirements.txt`
- Copies the inference API code into the image
- Exposes the API port
- Sets the default startup command

### Task 3 — GCP Service Account Setup
I created a GCP service account with the following roles:
- `Artifact Registry Writer` — to push Docker images
- `Kubernetes Engine Developer` — to deploy workloads to GKE

I then base64-encoded the service account key and stored it as a `GCP_SA_KEY` GitHub Actions secret, along with `GCP_PROJECT_ID`, `GKE_CLUSTER`, and `GKE_ZONE`.

### Task 4 — Build & Push via GitHub Actions
I created a GitHub Actions workflow (`.github/workflows/cd.yml`) that:
1. Authenticates with GCP using the service account secret
2. Configures Docker to push to Google Artifact Registry
3. Builds the Docker image
4. Pushes the image to Artifact Registry tagged with the commit SHA

### Task 5 — Deploy to GKE
I extended the workflow to:
1. Get GKE credentials (`gcloud container clusters get-credentials`)
2. Apply `deployment.yaml` and `service.yaml` Kubernetes manifests
3. Update the deployment image to the newly pushed tag (`kubectl set image`)

The IRIS API was successfully deployed and accessible via the external IP assigned by the `LoadBalancer` service. I demonstrated a live prediction call in my screencast.

### Task 6 (Optional) — MLflow Model in the Container
I fetched the best model from the MLflow Model Registry during the Docker build step and bundled it directly into the image. The deployed API serves predictions from the packaged model without requiring runtime access to MLflow.

---

## Pipeline Flow

```
Code Push → CI (Tests Pass) → Docker Build → Push to Artifact Registry → Deploy to GKE → Live API
```

---

## Project Structure

```
week_6/
├── .github/
│   └── workflows/
│       └── cd.yml          # CD pipeline (build, push, deploy)
├── src/
│   └── prepare.py              # data preparation file
│   └── train.py                # training file
├── iris_inference.py       # File containing fastapi app
├── k8s/
│   ├── deployment.yaml     # Kubernetes Deployment manifest
│   └── service.yaml        # Kubernetes LoadBalancer Service
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Learnings

- Containerizing an ML inference API with Docker and managing layer caching for faster builds
- Configuring GCP service accounts with least-privilege roles for CI/CD
- Writing Kubernetes manifests to declaratively manage deployments
- Automating the full build → push → deploy cycle with GitHub Actions
- Understanding how Kubernetes Services expose Pods to external traffic via a stable IP